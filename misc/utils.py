import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from importlib import reload
import sys
from types import ModuleType
import os


def decode_sequence(ix_to_word, seq):
    '''
    Input: seq is a tensor of size (batch_size, seq_length), with element 0 .. vocab_size. 0 is <END> token.
    '''
    if isinstance(seq, list):
        out = []
        for i in range(len(seq)):
            txt = ''
            for j in range(len(seq[i])):
                ix = seq[i][j]
                if ix > 0:
                    if j >= 1:
                        txt = txt + ' '
                    txt = txt + ix_to_word[str(ix)]
                else:
                    break
            out.append(txt)
        return out
    else:
        N, D = seq.size()
        out = []
        for i in range(N):
            txt = ''
            for j in range(D):
                ix = seq[i, j]
                if ix > 0:
                    if j >= 1:
                        txt = txt + ' '
                    txt = txt + ix_to_word[str(ix)]
                else:
                    break
            out.append(txt)
        return out


def decode_story(id2word, result):
    """
    :param id2word: vocab
    :param result: (batch_size, story_size, seq_length)
    :return:
    out: a list of stories. the size of the list is batch_size
    """
    batch_size, story_size, seq_length = result.size()
    out = []
    indexed_out = []
    for i in range(batch_size):
        sents = []
        txt = ''
        for j in range(story_size):
            sent = []
            for k in range(seq_length):
                vocab_id = result[i, j, k].item()
                if vocab_id > 0:
                    sent.append((k, id2word[str(vocab_id)]))
                    txt = txt + ' ' + id2word[str(vocab_id)]
                else:
                    break
            sents.append(sent)
        indexed_out.append(sents)
        out.append(txt)
    return out, indexed_out

def post_process_story(id2word, result):
    """
    :param id2word: vocab
    :param result: (batch_size, beam_size, story_size, seq_length)
    :return:
    out: a list of stories. the size of the list is batch_size
    """
    batch_size, story_size, beam_size, seq_length = result.shape
    out = []
    for i in range(batch_size):
        txts = []
        stories = []
        for j in range(story_size):
            for b in range(beam_size):
                txt = ''
                for k in range(seq_length):
                    vocab_id = result[i, j, b, k]
                    if vocab_id > 0:
                        txt = txt + ' ' + id2word[str(vocab_id)]
                    else:
                        break
            stories.append(txt)
        txts.append(stories)
        out.append(txts)
    return out


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def setup_seed():
    torch.manual_seed(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(12345)
    np.random.seed(12345)


def deep_reload(module, paths=None, mdict=None):
    """Recursively reload modules."""
    if paths is None:
        paths = ['']
    if mdict is None:
        mdict = {}
    if module not in mdict:
        # modules reloaded from this module
        mdict[module] = []
    reload(module)
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if type(attribute) is ModuleType:
            if attribute not in mdict[module]:
                if attribute.__name__ not in sys.builtin_module_names:
                    if os.path.dirname(attribute.__file__) in paths:
                        mdict[module].append(attribute)
                        deep_reload(attribute, paths, mdict)
    reload(module)
    #return mdict

def create_occurance_matrix(m, vocab_size):
    counter = np.zeros((m.shape[0],vocab_size), dtype=np.float32)
    for i in range(m.shape[0]):
        ind, count = np.unique(m[i], return_counts=True)
        counter[i][ind] = count
    return counter