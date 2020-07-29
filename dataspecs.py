import os

sep = os.sep
AV_WIDE = {
    'data_dir': 'AV-WIDE' + sep + 'images',
    'label_dir': 'AV-WIDE' + sep + 'manual',
    'split_dir': 'AV-WIDE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_vessels.png'
}
VEVIO = {
    'data_dir': 'VEVIO' + sep + 'mosaics',
    'label_dir': 'VEVIO' + sep + 'mosaics_manual_01_bw',
    'mask_dir': 'VEVIO' + sep + 'mosaics_masks',
    'split_dir': 'VEVIO' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_gt.png',
    'mask_getter': lambda file_name: 'mask_' + file_name
}
