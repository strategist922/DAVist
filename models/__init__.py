"""
adapted from https://github.com/eric-xw/AREL.git
"""
import os
import copy

import numpy as np
import torch

from .BaseModel import BaseModel
import logging
from six.moves import cPickle

def setup(opt):
    if opt.model == 'BaseModel':
        if opt.option == 'test':
            model_dir = os.path.join(opt.checkpoint_path, opt.id)
            with open(os.path.join(model_dir, 'infos-best.pkl'), 'rb') as f:
                info_best = cPickle.load(f)
            model = BaseModel(info_best['opt'])
        else:
            model = BaseModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.model))

    # check compatibility if training is continued from previously saved model
    if opt.option == 'test':
        assert os.path.isdir(model_dir), "{} must be a a path".format(model_dir)
        if opt.test_iter is None:
            model.load_state_dict(torch.load(os.path.join(model_dir, 'model-best.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(model_dir, f'model_iter_{opt.test_iter}.pth')))
    elif opt.resume_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.resume_from), "{} must be a a path".format(opt.resume_from)
        assert os.path.isfile(
            os.path.join(opt.resume_from, "infos.pkl")), "infos.pkl file does not exist in path {}".format(
            opt.resume_from)
        if opt.option == "test":
            print("Load the best model from {}".format(opt.resume_from))

        else:
            logging.info("Load pretrained model")
            model.load_state_dict(torch.load(os.path.join(opt.resume_from, 'model.pth')))
    elif opt.start_from_model is not None:
        if os.path.exists(opt.start_from_model):
            logging.info("Start from pretrained model")
            model.load_state_dict(torch.load(opt.start_from_model))
        else:
            err_msg = "model path doesn't exist: {}".format(opt.start_from_model)
            logging.error(err_msg)
            raise Exception(err_msg)

    return model
