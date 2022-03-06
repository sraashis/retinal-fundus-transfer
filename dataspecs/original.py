import os

sep = os.sep

"""--------------------------------------------------------------------------"""


def get_label_drive(file_name):
    return file_name.split('_')[0] + '_manual1.gif'


def get_mask_drive(file_name):
    return file_name.split('_')[0] + '_mask.gif'


DRIVE = {
    'name': 'DRIVE',
    'patch_shape': (388, 388),
    'patch_offset': (300, 300),
    'expand_by': (184, 184),
    'data_dir': 'DRIVE' + sep + 'images',
    'label_dir': 'DRIVE' + sep + 'manual',
    'split_dir': 'DRIVE' + sep + 'splits',
    'mask_dir': 'DRIVE' + sep + 'mask',
    'label_getter': get_label_drive,
    'mask_getter': get_mask_drive
}

"""--------------------------------------------------------------------"""


def get_labels_stare(file_name):
    return file_name.split('.')[0] + '.ah.pgm'


STARE = {
    'name': 'STARE',
    'patch_shape': (388, 388),
    'patch_offset': (300, 300),
    'expand_by': (184, 184),
    'data_dir': 'STARE' + sep + 'stare-images',
    'label_dir': 'STARE' + sep + 'labels-ah',
    'split_dir': 'STARE' + sep + 'splits',
    'label_getter': get_labels_stare,
}

"""--------------------------------------------------------------------------"""


def get_label_wide(file):
    return file.split('.')[0] + '_vessels.png'


AV_WIDE = {
    'name': 'WIDE',
    'patch_shape': (388, 388),
    'patch_offset': (300, 300),
    'expand_by': (184, 184),
    'data_dir': 'AV-WIDE' + sep + 'images',
    'label_dir': 'AV-WIDE' + sep + 'manual',
    'split_dir': 'AV-WIDE' + sep + 'splits',
    'label_getter': get_label_wide
}

"""--------------------------------------------------------------------------"""


def get_label_chasedb(file):
    return file.split('.')[0] + '_1stHO.png'


CHASEDB = {
    'name': 'CHASEDB',
    'patch_shape': (388, 388),
    'patch_offset': (300, 300),
    'expand_by': (184, 184),
    'data_dir': 'CHASEDB' + sep + 'images',
    'label_dir': 'CHASEDB' + sep + 'manual',
    'split_dir': 'CHASEDB' + sep + 'splits',
    'label_getter': get_label_chasedb
}

"""--------------------------------------------------------------------------"""


def get_label_HRF(file_name):
    return file_name.split('.')[0] + '.tif'


def get_mask_HRF(file_name):
    return file_name.split('.')[0] + '_mask.tif'


HRF = {
    'name': 'HRF',

    'patch_shape': (836, 836),
    'patch_offset': (650, 650),
    'expand_by': (184, 184),
    'data_dir': 'HRF' + sep + 'images',
    'label_dir': 'HRF' + sep + 'manual',
    'mask_dir': 'HRF' + sep + 'mask',
    'split_dir': 'HRF' + sep + 'splits',
    'label_getter': get_label_HRF,
    'mask_getter': get_mask_HRF
}

"""--------------------------------------------------------------------------"""


def get_label_iostar(file_name):
    return file_name.split('.')[0] + '_GT.tif'


def get_mask_iostar(file_name):
    return file_name.split('.')[0] + '_Mask.tif'


IOSTAR = {
    'name': 'IOSTAR',
    'patch_shape': (388, 388),
    'patch_offset': (300, 300),
    'expand_by': (184, 184),
    'data_dir': 'IOSTAR' + sep + 'image',
    'label_dir': 'IOSTAR' + sep + 'Vessel_GT',
    'mask_dir': 'IOSTAR' + sep + 'mask',
    'split_dir': 'IOSTAR' + sep + 'splits',
    'label_getter': get_label_iostar,
    'mask_getter': get_mask_iostar
}
