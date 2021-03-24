from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import time
import logging

import opts
from dataset import VISTDataset
import models
from log_utils import Logger
import misc.utils as utils

from eval_utils import Evaluator
import criterion
from misc.yellowfin import YFOptimizer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def setup_optimizer(opt, model):
    if opt.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=opt.learning_rate,
                               betas=(opt.optim_alpha, opt.optim_beta),
                               eps=opt.optim_epsilon,
                               weight_decay=opt.weight_decay)
    elif opt.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=opt.learning_rate,
                              momentum=opt.momentum)
    elif opt.optim == "momSGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=opt.learning_rate,
                                    momentum=opt.momentum)
    elif opt.optim == 'Adadelta':
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=opt.learning_rate,
                                   weight_decay=opt.weight_decay)
    elif opt.optim == 'RMSprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    else:
        logging.error("Unknown optimizer: {}".format(opt.optim))
        raise Exception("Unknown optimizer: {}".format(opt.optim))

    # Load the optimizer
    if opt.resume_from is not None:
        optim_path = os.path.join(opt.resume_from, "optimizer.pth")
        if os.path.isfile(optim_path):
            logging.info("Load optimizer from {}".format(optim_path))
            optimizer.load_state_dict(torch.load(optim_path))
            opt.learning_rate = optimizer.param_groups[0]['lr']
            logging.info("Loaded learning rate is {}".format(opt.learning_rate))

    return optimizer


def train(opt):
    # utils.setup_seed()
    logger = Logger(opt, save_code=opt.save_code)

    ################### set up dataset and dataloader ########################
    dataset = VISTDataset(opt)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()

    dataset.set_option(data_type={'whole_story': False, 'split_story': True, 'caption': False, 'prefix_story': True})

    dataset.train()
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)
    dataset.val()
    val_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    ##################### set up model, criterion and optimizer ######
    bad_valid = 0

    # set up evaluator
    evaluator = Evaluator(opt, 'val')

    # set up criterion
    crit = criterion.LanguageModelCriterion()

    # set up model
    model = models.setup(opt)
    model.cuda()

    # set up optimizer
    optimizer = setup_optimizer(opt, model)

    dataset.train()
    model.train()
    initial_lr = opt.learning_rate
    logging.info(model)
    ############################## training ##################################
    for epoch in range(logger.epoch_start, opt.max_epochs):
        # Assign the scheduled sampling prob
        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob *
                              frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob

        for iter, batch in enumerate(train_loader):
            start = time.time()
            logger.iteration += 1
            torch.cuda.synchronize()

            feature_fc = batch['feature_fc'].cuda()
            if opt.use_obj:
                feature_obj = batch['feature_obj'].cuda()
                if opt.use_spatial:
                    feature_obj_spatial = batch['feature_obj_spatial'].cuda()
                else:
                    feature_obj_spatial = None
                if opt.use_classes:
                    feature_obj_classes = batch['feature_obj_classes'].cuda()
                else:
                    feature_obj_classes = None
                if opt.use_attrs:
                    feature_obj_attrs = batch['feature_obj_attrs'].cuda()
                else:
                    feature_obj_attrs = None
            target = batch['split_story'].cuda()
            prefix = batch['prefix_story'].cuda()
            history_count = batch['history_counter'].cuda()
            index = batch['index']

            optimizer.zero_grad()

            # cross entropy loss
            output = model(feature_fc, feature_obj, target, history_count, spatial=feature_obj_spatial,
                               clss=feature_obj_classes, attrs=feature_obj_attrs)
            loss = crit(output, target)

            loss.backward()
            train_loss = loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip, norm_type=2)
            optimizer.step()
            torch.cuda.synchronize()

            if iter % opt.log_step == 0:
                logging.info("Epoch {} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, iter,
                                                                                                  len(train_loader),
                                                                                                  train_loss,
                                                                                                  time.time() - start))
            # Write the training loss summary
            if logger.iteration % opt.losses_log_every == 0:
                logger.log_training(epoch, iter, train_loss, opt.learning_rate, model.ss_prob)

            if logger.iteration % opt.save_checkpoint_every == 0:
                # Evaluate on validation dataset and save model for every epoch
                val_loss, predictions, metrics = evaluator.eval_story(model, crit, dataset, val_loader, opt)
                if opt.metric == 'XE':
                    score = -val_loss
                else:
                    score = metrics[opt.metric]
                logger.log_checkpoint(epoch, val_loss, metrics, predictions, opt, model, dataset, optimizer)
                # halve the learning rate if not improving for a long time
                if logger.best_val_score > score:
                    bad_valid += 1
                    if bad_valid >= opt.bad_valid_threshold:
                        opt.learning_rate = opt.learning_rate * opt.learning_rate_decay_rate
                        logging.info("halve learning rate to {}".format(opt.learning_rate))
                        checkpoint_path = os.path.join(logger.log_dir, 'model-best.pth')
                        model.load_state_dict(torch.load(checkpoint_path))
                        utils.set_lr(optimizer, opt.learning_rate)  # set the decayed rate
                        bad_valid = 0
                        logging.info("bad valid : {}".format(bad_valid))
                else:
                    opt.learning_rate = initial_lr
                    logging.info("achieving best {} score: {}".format(opt.metric, score))
                    bad_valid = 0


def test(opt):
    logger = Logger(opt)
    utils.setup_seed()
    dataset = VISTDataset(opt)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()

    dataset.test()
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    evaluator = Evaluator(opt, 'test')
    model = models.setup(opt)
    model.cuda()
    if opt.challenge:
        evaluator.test_challange(model, dataset, test_loader, opt)
    else:
        predictions, metrics = evaluator.test_story(model, dataset, test_loader, opt)


if __name__ == "__main__":
    opt = opts.parse_opt()
    if opt.option == 'train':
        print('Begin training:')
        train(opt)
    else:
        print('Begin testing:')
        test(opt)
