from easytorch import EasyTorch
from classification import MyTrainer, MyDataset
import os

sep = os.sep


def get_label_drive(file_name):
    return file_name.split('_')[0] + '_manual1.gif'


def get_mask_drive(file_name):
    return file_name.split('_')[0] + '_mask.gif'


DRIVE = {
    'name': 'DRIVE',
    'data_dir': 'DRIVE' + sep + 'images',
    'label_dir': 'DRIVE' + sep + 'manual',
    'mask_dir': 'DRIVE' + sep + 'mask',
    'label_getter': get_label_drive,
    'mask_getter': get_mask_drive
}



# STARE = {
#     'name': 'STARE',
#     'data_dir': 'STARE' + sep + 'stare-images',
#     'label_dir': 'STARE' + sep + 'labels-ah',
#     'label_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
# }

loader_args = {'train': {'batch_size': 2, 'drop_last': True}}
runner = EasyTorch([DRIVE],
                   phase='train', batch_size=4, epochs=31,
                   load_sparse=True, num_channel=1, num_class=2,
                   model_scale=2, dataset_dir='datasets', seed=1,
                   verbose=True, dataloader_args=loader_args)

if __name__ == "__main__":
    runner.run(MyTrainer, MyDataset)
    # runner.run_pooled(MyTrainer, MyDataset)
