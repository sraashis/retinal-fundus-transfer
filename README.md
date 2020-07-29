### This is a working example of [easytorch](https://github.com/sraashis/easytorch). A quick and easy way to run pytorch based neural network experiments. 
### This example consist of retinal blood vessel segmentation on two datasets- DRIVE<sub>[1]</sub>, and STARE<sub>[2]</sub>. 
### We have shown a per-data experiment setup, and pooled version of all datasets in this repo
Note that one **MUST cite the original authors** if these dataset are used in your research (references at the end).

1. Initialize the **dataspecs.py** as follows. Non existing directories will be automatically created in the first run.
```python
import os

sep = os.sep
DRIVE = {
    'data_dir': 'DRIVE' + sep + 'images',
    'label_dir': 'DRIVE' + sep + 'manual',
    'split_dir': 'DRIVE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('_')[0] + '_manual1.gif',
    'mask_getter': lambda file_name: file_name.split('_')[0] + '_mask.gif'
}
STARE = {
    'data_dir': 'STARE' + sep + 'stare-images',
    'label_dir': 'STARE' + sep + 'labels-ah',
    'split_dir': 'STARE' + sep + 'splits',
    'label_getter': lambda file_name: file_name.split('.')[0] + '.ah.pgm',
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
* [Example](https://github.com/sraashis/easytorch/tree/master/example) DRIVE dataset has the following structure:
    * datasets/DRIVE/images/
    * datasets/DRIVE/manual (segmentation ground truth)
    * datasets/DRIVE/splits
    * datasets/DRIVE/masks
* **splits** directory should consist **k** splits for k-fold cross validation. 
* **splits** are json files that determines which files are for test, validation , and for test.
* We have a [K-folds creater utility](https://github.com/sraashis/easytorch/blob/master/easytorch/utils/datautils.py) to generate such folds. So, at the moment a user have to use it to create the splits and place them in splits directory.
* This is super helpful when working with cloud deployment/ or google colab. 

2. Override our custom dataloader(**ETDataset**) and implement each item parser as in the example.
3. Initialize our custom neural network trainer(**ETTrainer**) and implement logic for one iteration, how to save evaluation scores. Sometimes we want to save predictions as images and all so it is necessary. Initialize log headers. More in example.
4. Implement the entry point
```python
import argparse

import dataspecs as dspec
from easytorch.utils.defaultargs import ap
from easytorch.runs import run, pooled_run
from classification import MyTrainer, MyDataset

ap = argparse.ArgumentParser(parents=[ap], add_help=False)

dataspecs = [dspec.DRIVE, dspec.STARE]
if __name__ == "__main__":
    run(ap, dataspecs, MyTrainer, MyDataset)
    pooled_run(ap, dataspecs, MyTrainer, MyDataset)
```

##### Parameters used in **Training+Validation+Test**
    * $python main.py -p train -nch 1 -e 21 -b 8 -sp True -mxp True -r 1
##### To run **Only Test**
    * $python main.py -p test -nch 3 -e 3 -b 2 -sp True

We would like to highlight a very use full feature called dataset pooling. With such, one can easily run experiments by combining any number of datasets as :
* For that, we only need to write dataspecs.py for the dataset we want to pool.
* **run** method runs for all dataset separately  at a time.
* **pooled_run** pools all the dataset and runs experiments like in the example where we combine two datasets **[dspec.DRIVE, dspec.STARE]** internally creating a larger unified dataset and training on that.
### Results for DRIVE, STARE and pooled are in net_logs folder
* It should be trained more epochs to gets state of the art result. 
* Pretrained weights are not uploaded because of space issues.

1. DRIVE dataset logs.
    * Training log
        ![DRIVE training log](net_logs/DRIVE/DRIVE_training_log.png)
    * Validation log
        ![DRIVE training log](net_logs/DRIVE/DRIVE_validation_log.png)

2. We ran 5-fold cross validation for STARE dataset. The following are logs of the first fold.
    * Scores for each folds, and global(combining all folds) in net_logs/STARE/_global_test_scores.csv
    
            |Fold                |Precision|REcall|F1    |Accuracy|
            |--------------------|---------|------|------|--------|
            |STARE_0.json        |0.8869   |0.6588|0.756 |0.9612  |
            |STARE_1.json        |0.8273   |0.8144|0.8208|0.975   |
            |STARE_4.json        |0.7509   |0.8309|0.7889|0.9682  |
            |STARE_3.json        |0.7945   |0.7675|0.7808|0.9643  |
            |STARE_2.json        |0.8663   |0.8235|0.8444|0.9731  |
            |Global              |0.8237   |0.7745|0.7983|0.9684  |

    * Training log
        ![STARE fold_0 training log](net_logs/STARE/STARE_0_training_log.png)
    * Validation log
        ![STARE fold_0 validation log](net_logs/STARE/STARE_0_training_log.png)

3. Pooled version
    * Training log
        ![Pooled training log](net_logs/pooled/pooled_training_log.png)
    * Validation log
        ![Pooled validation log](net_logs/pooled/pooled_validation_log.png)
        
## References
1. DRIVE Dataset, J. Staal, M. Abramoff, M. Niemeijer, M. Viergever, and B. van Ginneken, “Ridge based vessel segmentation in color images of the retina,” IEEE Transactions on Medical Imaging 23, 501–509 (2004)
2. STARE Dataset, A. D. Hoover, V. Kouznetsova, and M. Goldbaum, “Locating blood vessels in retinal images by piecewise threshold
       probing of a matched filter response,” IEEE Transactions on Med. Imaging 19, 203–210 (2000)
3. Architecture used, O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in
    MICCAI, (2015)
4. Our paper on vessel segmentation:
    * [Link to arxiv](https://arxiv.org/abs/1903.07803)
    * [Dynamic Deep Networks for Retinal Vessel Segmentation](https://www.frontiersin.org/articles/10.3389/fcomp.2020.00035/abstract)