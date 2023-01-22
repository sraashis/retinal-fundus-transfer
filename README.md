### An easy way of doing transfer learning for color fundus images processing tasks.
* Like vessel segmentation, Aretery/Vein classification, and optic disk segementation, exudates, segmentation/detection, and many more.
* It uses special custom Image transforms that works well with retinal images.
* A detailed explanation on transfer learning for retinal vessel segmentation with publicly available dataset to any datasets that does not have ground truth vessel mask available as follows:

### Step 1:
**Install pytorch and torchvision from [official website](https://pytorch.org/), and run:**

```
pip install easytorch==3.4.9
```

### Step 2:
* It comes with specifications to do transfer learning in directory [dataspecs/transfer.py](./dataspecs/transfer.py).
* Example specification for DRIVE dataset:

```python
def get_label_drive(file_name):
        return file_name.split('_')[0] + '_manual1.gif'

def get_mask_drive(file_name):
    return file_name.split('_')[0] + '_mask.gif'

DRIVE = {
    'name': f'DRIVE',
    'patch_shape': (388, 388),
    'patch_offset': (300, 300),
    'expand_by': (184, 184),
    'data_dir': 'DRIVE' + sep + 'images',
    'label_dir': 'DRIVE' + sep + 'manual',
    'mask_dir': 'DRIVE' + sep + 'mask',
    'label_getter': get_label_drive,
    'mask_getter': get_mask_drive,
    'resize': (896, 896),
    'thr_manual': 50
}
```
* We use U-Net model which works on patches of images with sliding window.
* After resizing, we need to binarize the ground truth using `thr_manual` threshold.
* This repo also includes a target dataset DDR for which we don't have vessel segmentation available. [DDR specification](./dataspecs/target.py). We want to do transfer learning from existing public datasets.
* Different ways we can do transfer learning for vessel segmentation:
  * We can either train by resizing all public datasets to approx same size as DDR. Thats what the `resize` is for in the above spec.  
  * We can train public dataset in a smaller size(this makes sense if target dataset is too large and hard to process), and also resize the DDR dataset to same size.
  * We train by using data augmentation; Simply train a model with different sizes of public dataset. It can get complicated because we need to resize the ground truths as well. But worry not, this code handles all of that.

### Finally:
**Case 1:** Run a working example on DDR dataset using two datasets(DRIVE and WIDE) for transfer learning.

`python main.py -ph train -data datasets --training-datasets DRIVE STARE --target-datasets DDR_train -spl 0.75 0.25 0 -b 8 -nw 6 -lr 0.001 -e 501 -pat 101 -rcw True`

**Case 2:** Use more datasets as below.

`python main.py -ph train -data <path to your dataset> --training-datasets DRIVE CHASEDB HRF IOSTAR STARE --target-datasets DDR_train -spl 0.75 0.25 0 -b 8 -nw 6 -lr 0.001 -e 501 -pat 101 -rcw True`

* This code uses `easytorch` framework and inherits some default args. Consult [easytorch repo](https://github.com/sraashis/easytorch) for details. But worry not, I will explain each of these.
  * **-ph** train: specifies which phase like train, test(for inference).
  * **-data** datasets: Path to your datasets so that you can run this code anywhere your data is. Just need to point to your datasets(Check [datasets](./datasets)) folder for an example.
  * **--training-dataset** ... : Which datasets to use for transfer learning from the specifications in [dataspecs](./dataspecs) directory. A single model will be trained using these datasets.
  * **--target-datasets** ... : After getting the best model, which dataset to use it on to generate vessel segmentation results.
  * **-spl** 0.75 0.25 0 : Split ratio for training dataset in order train, validation, test. We dont need test data for this transfer learning. We need validation set to pick the best model.
  * **-nw** 6 : num of workers
  * **-lr** 0.001 : Learning rate
  * **-e** 501: Epochs
  * **-pat** 101: patience to stop training. If model does not improve in previous 101 epochs, stop the training.
  * **-rcw** True : Stochastic weights scheme to improve prediction on fainter vessels as detailed in paper below(Dynamic Deep Networks for Retinal Vessel Segmentation).
  

### All the best! Cheers! 🎉

#### Please star or cite if you find it useful.

```
@article{deepdyn_10.3389/fcomp.2020.00035,
	title        = {Dynamic Deep Networks for Retinal Vessel Segmentation},
	author       = {Khanal, Aashis and Estrada, Rolando},
	year         = 2020,
	journal      = {Frontiers in Computer Science},
	volume       = 2,
	pages        = 35,
	doi          = {10.3389/fcomp.2020.00035},
	issn         = {2624-9898}
}

@misc{2202.02382,
        Author = {Aashis Khanal and Saeid Motevali and Rolando Estrada},
        Title = {Fully Automated Tree Topology Estimation and Artery-Vein Classification},
        Year = {2022},
        Eprint = {arXiv:2202.02382},
}
```
#### Please cite the respective datasets below if you use them in any way..

1. DRIVE Dataset, J. Staal, M. Abramoff, M. Niemeijer, M. Viergever, and B. van Ginneken, “Ridge based vessel
   segmentation in color images of the retina,” IEEE Transactions on Medical Imaging 23, 501–509 (2004)
2. STARE Dataset, A. D. Hoover, V. Kouznetsova, and M. Goldbaum, “Locating blood vessels in retinal images by piecewise
   threshold probing of a matched filter response,” IEEE Transactions on Med. Imaging 19, 203–210 (2000)
3. CHASE DB: Fraz, M. M., Remagnino, P., Hoppe, A., Uyyanonvara, B., Rudnicka, A. R., Owen, C. G., & Barman, S. A. (2012). An ensemble classification-based approach applied to retinal blood vessel segmentation. IEEE transactions on bio-medical engineering, 59(9), 2538–2548. https://doi.org/10.1109/TBME.2012.2205687
4. HRF Dataset: Budai, A., Bock, R., Maier, A., Hornegger, J., & Michelson, G. (2013). Robust vessel segmentation in fundus images. International journal of biomedical imaging, 2013, 154860. https://doi.org/10.1155/2013/154860
5. IOSTAR Dataset: J. Zhang, B. Dashtbozorg, E. Bekkers, J. P. W. Pluim, R. Duits and B. M. ter Haar Romeny, "Robust Retinal Vessel Segmentation via Locally Adaptive Derivative Frames in Orientation Scores," in IEEE Transactions on Medical Imaging, vol. 35, no. 12, pp. 2631-2644, Dec. 2016, doi: 10.1109/TMI.2016.2587062.
6. AV-WIDE Dataset: Estrada, R., Tomasi, C., Schmidler, S. C., & Farsiu, S. (2015). Tree Topology Estimation. IEEE transactions on pattern analysis and machine intelligence, 37(8), 1688–1701. https://doi.org/10.1109/TPAMI.2014.2382116
7. DDR Dataset: Tao Li, Yingqi Gao, Kai Wang, Song Guo, Hanruo Liu, & Hong Kang (2019). Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening. Information Sciences, 501, 511 - 522.
8. Architecture used, O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image
   segmentation,” in MICCAI, (2015)
