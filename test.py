import os
import time
import sys
import logging

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from torch.utils.data import DataLoader


import opts
from log_utils import Logger

from eval_utils import Evaluator
import criterion
from misc.yellowfin import YFOptimizer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def test(opt):
    from dataset import VISTDataset
    import models

    logger = Logger(opt)
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
    opt.option = 'test'
    sys.path.insert(0, os.path.join(opt.checkpoint_path, opt.id, 'src'))
    # for penalty in range(1,5):
    #     opt.penalty = float(penalty)
    print(f'Start test with beam {opt.beam_size} and penalty {opt.penalty}')
    test(opt)
