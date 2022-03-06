import os

sep = os.sep

resize = (896, 896)

DDR_TRAIN = {
    "name": "DDR_train",
    'patch_shape': (388, 388),
    'patch_offset': (300, 300),
    'expand_by': (184, 184),
    "data_dir": "DDR" + sep + "DR_grading" + sep + "train",
    "extension": "jpg",
    "bbox_crop": True,
    'resize': resize,
    'thr_manual': 50
}

DDR_VALID = {
    "name": "DDR_valid",
    'patch_shape': (388, 388),
    'patch_offset': (300, 300),
    'expand_by': (184, 184),
    "data_dir": "DDR" + sep + "DR_grading" + sep + "valid",
    "extension": "jpg",
    "bbox_crop": True,
    'resize': resize,
    'thr_manual': 50
}

DDR_TEST = {
    "name": "DDR_test",
    'patch_shape': (388, 388),
    'patch_offset': (300, 300),
    'expand_by': (184, 184),
    "data_dir": "DDR" + sep + "DR_grading" + sep + "test",
    "extension": "jpg",
    "bbox_crop": True,
    'resize': resize,
    'thr_manual': 50
}
