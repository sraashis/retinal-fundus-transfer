## This is a working example of [easytorch](https://github.com/sraashis/easytorch). A quick and easy way to run pytorch based neural network experiments. This example , consist of retinal blood vessel segmentation on two datasets. We have shown a per-data experiment setup, and pooled version of all datasets.

1. Initialize the **dataspecs.py** as follows. Non existing directories will be automatically created in the first run.
```python
import os

sep = os.sep
# --------------------------------------------------------------------------------------------

DRIVE = {
    'data_dir': 'DRIVE' + sep + 'images',
    'mask_dir': 'DRIVE' + sep + 'mask',
    'label_dir': 'DRIVE' + sep + 'OD_Segmentation',
    'split_dir': 'DRIVE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_gt.tif',
    'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif',
}

AV_WIDE = {
    'data_dir': 'AV-WIDE' + sep + 'images',
    'label_dir': 'AV-WIDE' + sep + 'OD_Segmentation',
    'split_dir': 'AV-WIDE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '_gt.png'
}
```
* **data_dir** is the path to images/or any data points.
* **label_dir** is the path to ground truth.
* **mask_dir** is the path to masks if any.
* **label_getter** is a function that gets corresponding ground truth of an image/data-point from **label_dir**.
* **mask_getter** is a function that gets corresponding mask of an image/data-point from **mask_dir**.

##### Please check [Our rich argument parser](https://github.com/sraashis/easytorch/blob/master/easytorch/utils/defaultargs.py)
* One of the arguments is -data/--dataset_dir which points to the root directory of the dataset. 
* So the program looks for an image say. image_001.png in dataset_dir/data_dir/images/image_001.png.
* [Example](https://github.com/sraashis/easytorch/tree/master/example) AV-WIDE dataset has the following structure:
    * datasets/AV-WIDE/images/
    * datasets/AV-WIDE/manual (segmentation ground truth)
    * datasets/AV-WIDE/splits
* **splits** directory should consist **k** splits for k-fold cross validation. 
* **splits** are json files that determines which files are for test, validation , and for test.
* We have a [K-folds creater utility](https://github.com/sraashis/easytorch/blob/master/easytorch/utils/datautils.py) to generate such folds. So, at the moment a user have to use it to create the splits and place them in splits directory.
* This is super helpful when working with cloud deployment/ or google colab. 

2. Override our custom dataloader(**QNDataset**) and implement each item parser as in the example.
3. Initialize our custom neural network trainer(**QNTrainer**) and implement logic for one iteration, how to save evaluation scores. Sometimes we want to save predictions as images and all so it is necessary. Initialize log headers. More in example.
4. Implement the entry point
```python
import argparse

import dataspecs as dspec
from easytorch.utils.defaultargs import ap
from easytorch.runs import run, pooled_run
from classification import MyTrainer, MyDataset

ap = argparse.ArgumentParser(parents=[ap], add_help=False)

dataspecs = [dspec.AV_WIDE, dspec.VEVIO]
if __name__ == "__main__":
    run(ap, dataspecs, MyTrainer, MyDataset)
    pooled_run(ap, dataspecs, MyTrainer, MyDataset)
```

##### **Training+Validation+Test**
    * $python main.py -p train -nch 3 -e 3 -b 2 -sp True
##### **Only Test**
    * $python main.py -p test -nch 3 -e 3 -b 2 -sp True

Here we like to highlight a very use ful feature call dataset pooling. With such, one can easily run experiments by combining any number of datasets as :
* For that, we only need to write dataspecs.py for the dataset we want to pool.
* **run** method runs for all dataset separately  at a time.
* **pooled_run** pools all the dataset and runs experiments like in the example where we combine two datasets **[dspec.DRIVE, dspec.AV_WIDE]** internally creating a larger unified dataset and training on that.





**Fundus images/masks used in example are from the following datasets. Whereas, optic disc ground truth are product of our work (Optical Disc Segmentation using Disk Centered Patch Augmentation):**
* AV-WIDE Dataset Reference:
    * R. Estrada, C. Tomasi, S. C. Schmidler, and S. Farsiu, “Tree topology estimation,” IEEE Transactions on Pattern
    Analysis Mach. Intell. 37, 1688–1701 (2015)
    * R. Estrada, M. J. Allingham, P. S. Mettu, S. W. Cousins, C. Tomasi, and S. Farsiu, “Retinal artery-vein classification
        via topology estimation,” IEEE Transactions on Med. Imaging 34, 2518–2534 (2015).
* VEVIO Dataset Reference: 
    * R. Estrada, C. Tomasi, M. T. Cabrera, D. K. Wallace, S. F. Freedman, and S. Farsiu, “Exploratory dijkstra forest
    based automatic vessel segmentation: applications in video indirect ophthalmoscopy (vio),” Biomed Opt Express 3,
    327–339 (2012). 22312585[pmid].
* Vessel Segmentation ground are a product of the following paper:
    * [Dynamic Deep Networks for Retinal Vessel Segmentation](https://www.frontiersin.org/articles/10.3389/fcomp.2020.00035/abstract)