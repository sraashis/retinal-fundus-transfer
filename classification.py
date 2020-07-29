import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tmf
from PIL import Image as IMG

from easytorch.utils.imageutils import (Image, get_chunk_indexes, expand_and_mirror_patch, merge_patches)
from easytorch.core.measurements import Avg, Prf1a
from easytorch.core.nn import ETTrainer, ETDataset
from models import UNet

sep = os.sep


class MyDataset(ETDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.get_label = kw.get('label_getter')
        self.get_mask = kw.get('mask_getter')
        self.patch_shape = (388, 388)
        self.patch_offset = (200, 200)
        self.input_shape = (572, 572)
        self.expand_by = (184, 184)
        self.image_objs = {}

    def load_index(self, map_id, file_id, file):
        dt = self.dmap[map_id]
        img_obj = Image()
        img_obj.load(dt['data_dir'], file)
        img_obj.load_ground_truth(dt['label_dir'], dt['label_getter'])
        img_obj.apply_clahe()
        img_obj.array = img_obj.array[:, :, 1]
        self.image_objs[file_id] = img_obj
        for corners in get_chunk_indexes(img_obj.array.shape, self.patch_shape, self.patch_offset):
            self.indices.append([map_id, file_id, file] + corners)

    def __getitem__(self, index):
        map_id, file_id, file, row_from, row_to, col_from, col_to = self.indices[index]

        img = self.image_objs[file_id].array
        gt = self.image_objs[file_id].ground_truth[row_from:row_to, col_from:col_to]

        p, q, r, s, pad = expand_and_mirror_patch(img.shape, [row_from, row_to, col_from, col_to], self.expand_by)
        img = np.pad(img[p:q, r:s], pad, 'reflect')

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img = np.flip(img, 0)
            gt = np.flip(gt, 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img = np.flip(img, 1)
            gt = np.flip(gt, 1)

        img = self.transforms(img)
        gt = self.transforms(gt)
        return {'indices': self.indices[index], 'input': img, 'label': gt.squeeze()}

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.ToPILImage(), tmf.ToTensor()])


class MyTrainer(ETTrainer):
    def __init__(self, args):
        super().__init__(args)

    def _init_nn(self):
        self.nn['model'] = UNet(self.args['num_channel'], self.args['num_class'], reduce_by=self.args['model_scale'])

    def iteration(self, batch):
        inputs = batch['input'].to(self.nn['device']).float()
        labels = batch['label'].to(self.nn['device']).long()

        out = self.nn['model'](inputs)
        loss = F.cross_entropy(out, labels)
        out = F.softmax(out, 1)

        _, pred = torch.max(out, 1)
        sc = self.new_metrics()
        sc.add(pred, labels)

        avg = Avg()
        avg.add(loss.item(), len(inputs))

        return {'loss': loss, 'avg_loss': avg, 'output': out, 'scores': sc, 'predictions': pred}

    def save_predictions(self, accumulator):
        """
        This only works if load_sparse is on. Which basically means patches of one image in one dataloader.
        At the moment, it works only if len(dataset) in less that the batch size.
        The implementation if it is greater than batch size is yet to be done, and can be done
         because all the necessary information is available here.
        """
        try:
            dataset_name = list(accumulator[0].dmap.keys()).pop()
            file = accumulator[1][0]['indices'][2][0].split('.')[0]
            out = accumulator[1][1]['output']
            img_shape = list(accumulator[0].image_objs.values())[0].array.shape
            patches = out[:, 1, :, :].cpu().numpy() * 255
            patches = np.array(patches.squeeze(), dtype=np.uint8)
            img = merge_patches(patches, img_shape, accumulator[0].patch_shape, accumulator[0].patch_offset)
            IMG.fromarray(img).save(self.cache['log_dir'] + sep + dataset_name + '_' + file + '.png')
        except:
            pass

    def new_metrics(self):
        return Prf1a()

    def reset_dataset_cache(self):
        self.cache['global_test_score'] = []
        self.cache['monitor_metrics'] = 'f1'
        self.cache['score_direction'] = 'maximize'

    def reset_fold_cache(self):
        self.cache['training_log'] = ['Loss,Precision,Recall,F1,Accuracy']
        self.cache['validation_log'] = ['Loss,Precision,Recall,F1,Accuracy']
        self.cache['test_score'] = ['Split,Precision,Recall,F1,Accuracy']
        self.cache['best_score'] = 0.0
