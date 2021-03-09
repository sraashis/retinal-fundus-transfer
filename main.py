from easytorch import EasyTorch
from classification import MyTrainer, MyDataset
import os

sep = os.sep

DRIVE = {
    'name': 'DRIVE',
    'data_dir': 'DRIVE' + sep + 'images',
    'label_dir': 'DRIVE' + sep + 'manual',
    'mask_dir': 'DRIVE' + sep + 'mask',
    'label_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
    'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
}
STARE = {
    'name': 'STARE',
    'data_dir': 'STARE' + sep + 'stare-images',
    'label_dir': 'STARE' + sep + 'labels-ah',
    'label_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
}

loader_args = {'train': {'batch_size': 2, 'drop_last': True}}
runner = EasyTorch([DRIVE, STARE],
                   phase='train', batch_size=4, epochs=2,
                   load_sparse=True, num_channel=1, num_class=2,
                   model_scale=8, dataset_dir='datasets',seed=1, load_limit=15,
                   verbose=True, num_folds=3, dataloader_args=loader_args)

if __name__ == "__main__":
    runner.run(MyTrainer, MyDataset)
    # runner.run_pooled(MyTrainer, MyDataset)
