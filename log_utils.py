"""
adapted from https://github.com/eric-xw/AREL.git
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

import torch
import numpy as np
import time
import os
from six.moves import cPickle
import logging


class TensorBoard:
    def __init__(self, opt):
        self.dir = os.path.join(opt.checkpoint_path, 'tensorboard', opt.id)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if opt.option != 'train' or opt.no_tb:
            logging.info(f"No tensorboard in this mode.")
            self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.dir, comment=f' opt.id')
            logging.info(f'logging to tensorboard at {self.dir}')
        except ImportError:
            logging.info("Tensorflow not installed; No tensorboard logging.")
            self.writer = None


class Logger:
    def __init__(self, opt, save_code=False):
        self.start = time.time()  # start timing

        self.log_dir = os.path.join(opt.checkpoint_path, opt.id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # set up logging
        self.set_logging(self.log_dir, opt)

        self.tensorboard = TensorBoard(opt)

        # print all the options
        logging.info("Option settings:")
        for k, v in vars(opt).items():
            if k == 'vocab': continue
            logging.info("{:40}: {}".format(k, v))

        if save_code:
            self.code_dir = os.path.join(self.log_dir, 'src')
            if not os.path.exists(self.code_dir):
                os.makedirs(self.code_dir)
            for file in os.listdir('.'):
                name = file.split('.')[0]
                if file.endswith('.py') and \
                        any([name == n for n in ['criterion', 'dataset', 'opts', 'train', 'eval_utils']]):
                    shutil.copyfile(file, os.path.join(self.code_dir, file))
            model_path = os.path.join(self.code_dir, 'models')
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            shutil.copytree('./models', model_path)

        self.infos = {}
        self.histories = {}
        if opt.resume_from is not None:
            # open old infos and check if models are compatible
            with open(os.path.join(opt.resume_from, 'infos.pkl'), 'rb') as f:
                self.infos = cPickle.load(f)
                saved_model_opt = self.infos['opt']
                need_be_same = ["num_layers"]
                for checkme in need_be_same:
                    assert vars(saved_model_opt)[checkme] == vars(opt)[
                        checkme], "Command line argument and saved model disagree on {}".format(checkme)

            if os.path.isfile(os.path.join(opt.resume_from, 'histories.pkl')):
                with open(os.path.join(opt.resume_from, 'histories.pkl'), 'rb') as f:
                    self.histories = cPickle.load(f)

        self.iteration = self.infos.get('iter', 0)  # total number of iterations, regardless epochs
        self.epoch_start = self.infos.get('epoch', -1) + 1
        if opt.load_best_score:
            self.best_val_score = self.infos.get('best_val_score', None)

        self.val_result_history = self.histories.get('val_result_history', {})
        self.loss_history = self.histories.get('loss_history', {})
        self.lr_history = self.histories.get('lr_history', {})
        self.ss_prob_history = self.histories.get('ss_prob_history', {})

    def set_logging(self, log_dir, opt):
        # set up logging to file - see previous section for more details
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=os.path.join(log_dir, f"{opt.option}_log.txt"),
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

    def log_training(self, epoch, iter, train_loss, current_lr, ss_prob):
        if self.tensorboard.writer is not None:
            self.tensorboard.writer.add_scalar('train_loss', train_loss, self.iteration)
            # self.tensorboard.writer.add_scalars('loss', {'train_loss': train_loss}, self.iteration)
            self.tensorboard.writer.add_scalar('learning_rate', current_lr, self.iteration)
            self.tensorboard.writer.add_scalar('scheduled_sampling_prob', ss_prob, self.iteration)
            self.tensorboard.writer.flush()

        self.loss_history[self.iteration] = train_loss
        self.lr_history[self.iteration] = current_lr
        self.ss_prob_history[self.iteration] = ss_prob

    def log_checkpoint(self, epoch, val_loss, metrics, predictions, opt, model, dataset, optimizer=None):
        # Write validation result into summary
        if self.tensorboard.writer is not None:
            self.tensorboard.writer.add_scalar('validation loss', val_loss, self.iteration)
            # self.tensorboard.writer.add_scalars('loss', {'validation loss': val_loss}, self.iteration)
            for k, v in metrics.items():
                self.tensorboard.writer.add_scalar(k, v, self.iteration)
                self.tensorboard.writer.flush()
        self.val_result_history[self.iteration] = {'loss': val_loss, 'metrics': metrics.copy(), 'predictions': predictions}

        # Save model if the validation result is improved
        if opt.metric == 'XE':
            current_score = -val_loss
        else:
            current_score = metrics[opt.metric]

        best_flag = False
        if self.best_val_score is None or current_score > self.best_val_score:
            self.best_val_score = current_score
            best_flag = True

        # save the model at current iteration
        checkpoint_path = os.path.join(self.log_dir, 'model_iter_{}.pth'.format(self.iteration))
        torch.save(model.state_dict(), checkpoint_path)
        # save as latest model
        checkpoint_path = os.path.join(self.log_dir, 'model.pth')
        torch.save(model.state_dict(), checkpoint_path)
        logging.info("model saved to {}".format(checkpoint_path))
        # save optimizer
        if optimizer is not None:
            optimizer_path = os.path.join(self.log_dir, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)

        # Dump miscalleous informations
        self.infos['iter'] = self.iteration
        self.infos['epoch'] = epoch
        self.infos['best_val_score'] = self.best_val_score
        self.infos['opt'] = opt
        self.infos['vocab'] = dataset.get_vocab()

        self.histories['val_result_history'] = self.val_result_history
        self.histories['loss_history'] = self.loss_history
        self.histories['lr_history'] = self.lr_history
        self.histories['ss_prob_history'] = self.ss_prob_history
        with open(os.path.join(self.log_dir, 'infos.pkl'), 'wb') as f:
            cPickle.dump(self.infos, f)
        with open(os.path.join(self.log_dir, 'histories.pkl'), 'wb') as f:
            cPickle.dump(self.histories, f)

        if best_flag:
            checkpoint_path = os.path.join(self.log_dir, 'model-best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info("model saved to {}".format(checkpoint_path))
            with open(os.path.join(self.log_dir, 'infos-best.pkl'), 'wb') as f:
                cPickle.dump(self.infos, f)
