import os

sep = os.sep


def get_label_drive(file_name):
    return file_name.split('_')[0] + '_manual1.gif'


def get_mask_drive(file_name):
    return file_name.split('_')[0] + '_mask.gif'


def get_label_iostar(file_name):
    return file_name.split('.')[0] + '_GT.tif'


def get_mask_iostar(file_name):
    return file_name.split('.')[0] + '_Mask.tif'


def get_label_wide(file):
    return file.split('.')[0] + '_vessels.png'


def get_label_chasedb(file):
    return file.split('.')[0] + '_1stHO.png'


def get_label_HRF(file_name):
    return file_name.split('.')[0] + '.tif'


def get_mask_HRF(file_name):
    return file_name.split('.')[0] + '_mask.tif'


def get_labels_stare(file_name):
    return file_name.split('.')[0] + '.ah.pgm'


def get(resize=(896, 896)):
    DRIVE = {
        'name': f'DRIVE{resize[0]}',
        'patch_shape': (388, 388),
        'patch_offset': (300, 300),
        'expand_by': (184, 184),
        'data_dir': 'DRIVE' + sep + 'images',
        'label_dir': 'DRIVE' + sep + 'manual',
        'mask_dir': 'DRIVE' + sep + 'mask',
        'label_getter': get_label_drive,
        'mask_getter': get_mask_drive,
        'resize': resize,
        'thr_manual': 50
    }

    STARE = {
        'name': f'STARE{resize[0]}',
        'patch_shape': (388, 388),
        'patch_offset': (300, 300),
        'expand_by': (184, 184),
        'data_dir': 'STARE' + sep + 'stare-images',
        'label_dir': 'STARE' + sep + 'labels-ah',
        'label_getter': get_labels_stare,
        'resize': resize,
        'thr_manual': 50
    }

    AV_WIDE = {
        'name': f'WIDE{resize[0]}',
        'patch_shape': (388, 388),
        'patch_offset': (300, 300),
        'expand_by': (184, 184),
        'data_dir': 'AV-WIDE' + sep + 'images',
        'label_dir': 'AV-WIDE' + sep + 'manual',
        'label_getter': get_label_wide,
        'resize': resize,
        'thr_manual': 50
    }

    CHASEDB = {
        'name': f'CHASEDB{resize[0]}',
        'patch_shape': (388, 388),
        'patch_offset': (300, 300),
        'expand_by': (184, 184),
        'data_dir': 'CHASEDB' + sep + 'images',
        'label_dir': 'CHASEDB' + sep + 'manual',
        'label_getter': get_label_chasedb,
        'resize': resize,
        'thr_manual': 50
    }

    HRF = {
        'name': f'HRF{resize[0]}',
        'patch_shape': (388, 388),
        'patch_offset': (300, 300),
        'expand_by': (184, 184),
        'data_dir': 'HRF' + sep + 'images',
        'label_dir': 'HRF' + sep + 'manual',
        'mask_dir': 'HRF' + sep + 'mask',
        'label_getter': get_label_HRF,
        'mask_getter': get_mask_HRF,
        'resize': resize,
        'thr_manual': 50
    }

    IOSTAR = {
        'name': f'IOSTAR{resize[0]}',
        'patch_shape': (388, 388),
        'patch_offset': (300, 300),
        'expand_by': (184, 184),
        'data_dir': 'IOSTAR' + sep + 'image',
        'label_dir': 'IOSTAR' + sep + 'Vessel_GT',
        'mask_dir': 'IOSTAR' + sep + 'mask',
        'label_getter': get_label_iostar,
        'mask_getter': get_mask_iostar,
        'resize': resize,
        'thr_manual': 50
    }
    return [DRIVE, STARE, AV_WIDE, CHASEDB, HRF, IOSTAR]
