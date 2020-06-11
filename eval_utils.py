# encoding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import json
import hashlib
import pandas as pd
import time
from vist_eval.album_eval import AlbumEvaluator
import logging
import subprocess

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import misc.utils as utils


class CocoResFormat:
    def __init__(self):
        self.res = []
        self.caption_dict = {}

    def read_multiple_files(self, filelist, hash_img_name):
        for filename in filelist:
            # print('In file %s\n' % filename)
            self.read_file(filename, hash_img_name)

    def read_file(self, filename, hash_img_name):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.split('\t')
                if len(id_sent) > 2:
                    id_sent = id_sent[-2:]
                assert len(id_sent) == 2
                sent = id_sent[1].strip()

                if hash_img_name:
                    img_id = int(int(hashlib.sha256(id_sent[0].encode('utf8')).hexdigest(),
                                     16) % sys.maxsize)
                else:
                    img_id = id_sent[0]
                imgid_sent = {}

                if img_id in self.caption_dict:
                    print(img_id)
                    assert self.caption_dict[img_id] == sent
                else:
                    self.caption_dict[img_id] = sent
                    imgid_sent['image_id'] = img_id
                    imgid_sent['caption'] = sent
                    self.res.append(imgid_sent)

    def dump_json(self, outfile):
        with open(outfile, 'w') as fd:
            json.dump(self.res, fd, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=(',', ': '))


class Evaluator:
    def __init__(self, opt, mode='val'):
        if opt.task == 'story_telling' or opt.task == 'story_telling_with_caption':
            ref_json_path = os.path.join(opt.reference_dir, "{}_reference.json".format(mode))
        else:
            ref_json_path = os.path.join(opt.reference_dir, "{}_desc_reference.json".format(mode))
        self.reference = json.load(open(ref_json_path))
        print("loading file {}".format(ref_json_path))
        self.save_dir = os.path.join(opt.checkpoint_path, opt.id)
        if mode == 'test':
            mode += '_best' if opt.test_iter is None else f'_{opt.test_iter}'
            mode += f'_b{opt.beam_size}' if opt.beam_size > 1 else f'_sample'
            mode += f'_p{opt.penalty}' if opt.penalty > 0 else ''
        self.prediction_file = os.path.join(self.save_dir, 'prediction_{}'.format(mode))
        self.eval = AlbumEvaluator(log=opt.log_metrics)

    def measure(self):
        json_prediction_file = '{}.json'.format(self.prediction_file)
        predictions = {}
        with open(self.prediction_file) as f:
            for line in f:
                vid, seq = line.strip().split('\t')
                if vid not in predictions:
                    predictions[vid] = [seq]
        self.eval.evaluate(self.reference, predictions)
        with open(json_prediction_file, 'w') as f:
            json.dump(predictions, f)
        return self.eval.eval_overall

    def eval_story(self, model, crit, dataset, loader, opt, side_model=None):
        # Make sure in the evaluation mode
        logging.info("Evaluating...")
        start = time.time()
        model.eval()
        dataset.val()

        loss_sum = 0
        loss_evals = 0
        predictions = {}

        prediction_txt = open(self.prediction_file, 'w')  # open the file to store the predictions

        count = 0
        with torch.no_grad():
            for iter, batch in enumerate(loader):
                iter_start = time.time()

                feature_fc = batch['feature_fc'].cuda()
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
                conv_feature = batch['feature_conv'].cuda() if 'feature_conv' in batch else None

                count += feature_fc.size(0)

                output = model(feature_fc, feature_obj, target, history_count, spatial=feature_obj_spatial,
                               clss=feature_obj_classes, attrs=feature_obj_attrs)

                loss = crit(output, target).item()
                loss_sum += loss
                loss_evals += 1

                # forward the model to also get generated samples for each video
                results, _ = model.predict(feature_fc, feature_obj, beam_size=opt.beam_size, penalty=opt.penalty,
                                           spatial=feature_obj_spatial, clss=feature_obj_classes,
                                           attrs=feature_obj_attrs, frequencies=dataset.frequency,
                                           function_words=dataset.get_function_words())
                stories, _ = utils.decode_story(dataset.get_vocab(), results)

                indexes = batch['index'].numpy()
                for j, story in enumerate(stories):
                    vid, _ = dataset.get_id(indexes[j])
                    if vid not in predictions:  # only predict one story for an album
                        # write into txt file for evaluate metrics like Cider
                        prediction_txt.write('{}\t {}\n'.format(vid, story))
                        # save into predictions
                        predictions[vid] = story

                logging.info("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(iter,
                                                                                    len(loader),
                                                                                    iter * 100.0 / len(loader),
                                                                                    time.time() - iter_start))

            prediction_txt.close()
            metrics = self.measure()  # compute all the language metrics

            # Switch back to training mode
            model.train()
            dataset.train()
            logging.info("Evaluation finished. Evaluated {} samples. Time used: {}".format(count, time.time() - start))
        return loss_sum / loss_evals, predictions, metrics

    def test_story(self, model, dataset, loader, opt):
        logging.info("Evaluating...")
        start = time.time()
        model.eval()
        dataset.test()

        predictions = {}
        prediction_txt = open(self.prediction_file, 'w')  # open the file to store the predictions

        with torch.no_grad():
            for iter, batch in enumerate(loader):
                iter_start = time.time()

                feature_fc = batch['feature_fc'].cuda()
                feature_conv = batch['feature_conv'].cuda() if 'feature_conv' in batch else None
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

                results, _ = model.predict(feature_fc, feature_obj, beam_size=opt.beam_size,
                                           spatial=feature_obj_spatial, clss=feature_obj_classes,
                                           attrs=feature_obj_attrs, penalty=opt.penalty,
                                           frequencies=dataset.frequency, function_words=dataset.get_function_words())

                sents, _ = utils.decode_story(dataset.get_vocab(), results)

                indexes = batch['index'].numpy()
                for j, story in enumerate(sents):
                    vid, _ = dataset.get_id(indexes[j])
                    if vid not in predictions:  # only predict one story for an album
                        # write into txt file for evaluate metrics like Cider
                        prediction_txt.write('{}\t {}\n'.format(vid, story))
                        # save into predictions
                        predictions[vid] = story

                print("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(iter,
                                                                             len(loader),
                                                                             iter * 100.0 / len(loader),
                                                                             time.time() - iter_start))

            prediction_txt.close()
            metrics = self.measure()  # compute all the language metrics

            json.dump(metrics, open(self.prediction_file.replace('prediction', 'scores'), 'w'))
            # Switch back to training mode
            print("Test finished. Time used: {}".format(time.time() - start))
        return predictions, metrics

    def test_challange(self, model, dataset, loader, opt, side_model=None):
        # Make sure in the evaluation mode
        logging.info("Evaluating...")
        start = time.time()
        model.eval()
        dataset.test()

        predictions = {"team_name": "", "evaluation_info": {"additional_description": ""}, "output_stories": []}

        prediction_txt = open(self.prediction_file, 'w')  # open the file to store the predictions

        count = 0
        finished_flickr_ids = []
        with torch.no_grad():
            for iter, batch in enumerate(loader):
                iter_start = time.time()

                feature_fc = batch['feature_fc'].cuda()
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
                count += feature_fc.size(0)
                results, _ = model.predict(feature_fc, feature_obj, beam_size=opt.beam_size, spatial=feature_obj_spatial,
                                           clss=feature_obj_classes, attrs=feature_obj_attrs, penalty=opt.penalty,
                                           frequencies=dataset.frequency, function_words=dataset.get_function_words())
                stories, _ = utils.decode_story(dataset.get_vocab(), results)

                indexes = batch['index'].numpy()
                for j, story in enumerate(stories):
                    album_id, flickr_id = dataset.get_all_id(indexes[j])
                    concat_flickr_id = "-".join(flickr_id)
                    if concat_flickr_id not in finished_flickr_ids:
                        # if vid not in predictions:  # only predict one story for an album
                        # write into txt file for evaluate metrics like Cider
                        prediction_txt.write('{}\t {}\n'.format(album_id, story))
                        # save into predictions
                        predictions['output_stories'].append(
                            {'album_id': album_id, 'photo_sequence': flickr_id, 'story_text_normalized': story})
                        finished_flickr_ids.append(concat_flickr_id)

                logging.info("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(iter,
                                                                                    len(loader),
                                                                                    iter * 100.0 / len(loader),
                                                                                    time.time() - iter_start))

            prediction_txt.close()
            json_prediction_file = os.path.join(self.save_dir, 'challenge.json')
            with open(json_prediction_file, 'w') as f:
                json.dump(predictions, f)

            logging.info("Evaluation finished. Evaluated {} samples. Time used: {}".format(count, time.time() - start))
            subprocess.call(["java", "-jar", opt.challenge_dir, "-testFile",
                             os.path.join(self.save_dir, 'challenge.json'),
                             "-gsFile", opt.sis_path])
        return predictions
