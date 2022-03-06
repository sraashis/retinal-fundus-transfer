import os
import random

import easytorch.vision.imageutils as imgutils
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tmf
from PIL import Image as IMG
from easytorch import ETTrainer, Prf1a, ETMeter
from easytorch.vision import (merge_patches)
from easytorch.vision.imgdataset2d import BinarySemSegImgPatchDataset
from easytorch.vision.transforms import RandomGaussJitter

from models import UNet

sep = os.sep


class VesselSegTrainer(ETTrainer):

    def _init_nn_model(self):
        self.nn['model'] = UNet(self.args['num_channel'], self.args['num_class'], reduce_by=self.args['model_scale'])

    def _init_optimizer(self):
        first_model = list(self.nn.keys())[0]
        self.optimizer['adam'] = torch.optim.Adam(self.nn[first_model].parameters(),
                                                  lr=self.args['learning_rate'])
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer['adam'],
        #     patience=35,
        #     min_lr=0.0001,
        #     cooldown=15,
        #     verbose=self.args.get('verbose')
        # )

    def iteration(self, batch, **kw):
        r"""
        :param batch:
        :return: dict with keys - loss(computation graph), averages, output, metrics, predictions
        """
        inputs = batch['input'].to(self.device['gpu']).float()
        labels = batch['label'].to(self.device['gpu']).long()
        out = self.nn['model'](inputs)

        wt = None
        if self.args.get('random_class_weights') is not None:
            wt = torch.randint(1, 101, (self.args['num_class'],), device=self.device['gpu']).float()

        elif self.args.get('class_weights') is not None:
            wt = self.cache.setdefault('class_weights', torch.from_numpy(
                np.array(self.args.get('class_weights'))
            ).float().to(self.device['gpu']))

        loss = F.cross_entropy(out, labels, weight=wt)
        out = F.softmax(out, 1)

        _, pred = torch.max(out, 1)
        meter = self.new_meter()
        meter.averages.add(loss.item(), len(inputs))

        if self.args['num_class'] == 2:
            meter.metrics['prf1a'].add(pred, labels.float())
        else:
            meter.metrics['cfm'].add(pred, labels.float())

        return {'loss': loss, 'output': out, 'meter': meter, 'predictions': pred, 'labels': labels}

    def save_predictions(self, dataset, its):
        if not self.args.get('load_sparse'):
            return None

        """load_sparse option in default params loads patches of single image in one dataloader.
         This enables to merge them safely to form the whole image """
        dname, file, cache_key = dataset.indices[0][0], dataset.indices[0][1], dataset.indices[0][-1]
        dspec = dataset.dataspecs[dname]
        obj = dataset.diskcache.get(cache_key)
        """
        Auto gather all the predicted patches of one image and merge together by calling as follows."""
        img_shape = obj.array.shape[:2]
        patches = its['output']()[:, 1, :, :].cpu().numpy() * 255
        img = merge_patches(patches, img_shape, dspec['patch_shape'], dspec['patch_offset'])

        _dset_dir = self.cache['log_dir']
        if self.args.get('pooled_run'):
            _dset_dir = f"{_dset_dir}{sep}{dspec['name']}"
            os.makedirs(_dset_dir, exist_ok=True)

        IMG.fromarray(img).save(_dset_dir + sep + file.split('.')[0] + '.png')

        patches = its['predictions']().cpu().numpy() * 255
        pred = merge_patches(patches, img_shape, dspec['patch_shape'], dspec['patch_offset'])
        sc = Prf1a()
        sc.add(torch.Tensor(pred), torch.Tensor(obj.ground_truth))
        return ETMeter(prf1a=sc)

    def init_experiment_cache(self):
        self.cache.update(monitor_metric='f1', metric_direction='maximize')
        self.cache.update(log_header='Loss|Accuracy,F1,Precision,Recall')

    def new_meter(self):
        return ETMeter(
            prf1a=Prf1a()
        )

    # def _on_epoch_end(self, epoch, training_meter=None, validation_meter=None):
    #     self.lr_scheduler.step(validation_meter.extract(self.cache['monitor_metric']))


class BinarySemSegImgPatchDatasetCustomTransform(BinarySemSegImgPatchDataset):

    def get_transforms(self):
        if self.mode == "test":
            return tmf.Compose([tmf.ToPILImage(), tmf.ToTensor()])

        _tf = [
            tmf.ToPILImage(),
            tmf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            tmf.RandomAutocontrast(),
            RandomGaussJitter(0.3, 0.5),
            tmf.ToTensor()
        ]
        return tmf.Compose(_tf)

    def __getitem__(self, index):
        dname, file, row_from, row_to, col_from, col_to, cache_key = self.indices[index]

        obj = self.diskcache.get(cache_key)
        img = obj.array[:, :, 1]  # Only Green Channel
        gt = obj.ground_truth[row_from:row_to, col_from:col_to]

        p, q, r, s, pad = imgutils.expand_and_mirror_patch(
            img.shape,
            [row_from, row_to, col_from, col_to],
            self.dataspecs[dname]['expand_by']
        )
        if len(img.shape) == 3:
            pad = [*pad, (0, 0)]

        img = np.pad(img[p:q, r:s], pad, 'reflect')
        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img = np.flip(img, 0)
            gt = np.flip(gt, 0)

        if self.mode == 'train' and random.uniform(0, 1) <= 0.5:
            img = np.flip(img, 1)
            gt = np.flip(gt, 1)

        img = self.transforms(img)
        gt = self.pil_to_tensor(gt)
        return {'indices': self.indices[index], 'input': img, 'label': gt.squeeze()}
