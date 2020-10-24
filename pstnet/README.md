# PST900 Inference & Verification

### Note: 

This section of the repository is new, and there may be hiccups in getting everything to work. 

## Requirements

The main requirements are:

* PyTorch 1.0
* TorchVision
* PIL
* Scikit-Image

## Generating Table II

You can generate all the results from Table II of our paper by running the following script:

```
python3 generate_inference_table.py
```

Before you do that, make sure you have done the following:

* Downloaded the dataset (link in data/data folder)
```
cd data/data/
wget https://www.dropbox.com/s/5eyfi503gp5ri8d/PST900_RGBT_Dataset.zip
unzip PST900_RGBT_Dataset.zip
```
* Downloaded the pre-trained models (link in data/weights folder)
```
cd data/weights/
wget https://www.dropbox.com/s/nregozp2anygdjw/weights.zip
unzip weights.zip
```

If that dropbox link doesn't work, please try this Google Drive link - https://drive.google.com/file/d/1frRfClQQP4th-cjxJwKmBuMXxBw2CYdF/view?usp=sharing

* Organized the repository in the structure provided below

Additionally, you will need to provide paths to the *data* folder of this repository in the file:

```
pst_inference.py
```

Make changes to the following lines:
```
# Directory of model files
ARTIFACT_DETECTION_DIR = "/data/pst900_thermal_rgb/pstnet/data/"
# Directory which contains batch of input images
ARGS_INFERENCE_DIR = ARTIFACT_DETECTION_DIR + "data/PST900_RGBT_Dataset/test/"
```

If you wish to also evaluate FastSCNN, make the same update to the file:

```
pstnet/code/FastSCNN/eval.py
```

Make similar changes to the following lines:
```
# Directory of model files
ARTIFACT_DETECTION_DIR = "/data/pst900_thermal_rgb/pstnet/data/"
# Directory which contains batch of input images
ARGS_INFERENCE_DIR = ARTIFACT_DETECTION_DIR + "data/PST900_RGBT_Dataset/test/"
```

## Evaluation

Once the repository structure is the same as mentioned below, and you have made the necessary changes to the files mentioned above, you can generate the Table II results by running the following script:

```
python3 generate_inference_table.py
```

And you should see the following output for the 13 different network variants tested.

```
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/erfnet_rgb/erfnet_rgb
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/erfnet_rgb/model_best.pth
Model and weights loaded..
---------------------------------
=============== erfnet_rgb VALIDATION ==============
[Validation] mIOU                : 0.6185150112722362
[Validation] Background          : 0.9869230037888026
[Validation] Fire extinguisher   : 0.6118676157755413
[Validation] Backpack            : 0.6528128395348313
[Validation] Hand drill          : 0.4240574442147192
[Validation] Rescue randy        : 0.4169141530472867
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/erfnet/erfnet
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/erfnet/model_best.pth
Model and weights loaded..
---------------------------------
=============== erfnet VALIDATION ==============
[Validation] mIOU                : 0.6255247202316041
[Validation] Background          : 0.9873264010569058
[Validation] Fire extinguisher   : 0.5879817243576662
[Validation] Backpack            : 0.6807423684232022
[Validation] Hand drill          : 0.5277323243328909
[Validation] Rescue randy        : 0.34384078298735554
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/mavnet_rgb/mavnet_rgb
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/mavnet_rgb/model_best.pth
Model and weights loaded..
---------------------------------
=============== mavnet_rgb VALIDATION ==============
[Validation] mIOU                : 0.4555154191130019
[Validation] Background          : 0.982194930274054
[Validation] Fire extinguisher   : 0.28317333696376074
[Validation] Backpack            : 0.585316953356088
[Validation] Hand drill          : 0.33671494846612854
[Validation] Rescue randy        : 0.09017692650497793
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/mavnet/mavnet
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/mavnet/model_best.pth
Model and weights loaded..
---------------------------------
=============== mavnet VALIDATION ==============
[Validation] mIOU                : 0.47735902050711204
[Validation] Background          : 0.9789180596042251
[Validation] Fire extinguisher   : 0.22585401295001736
[Validation] Backpack            : 0.5152457022089879
[Validation] Hand drill          : 0.3194607394816177
[Validation] Rescue randy        : 0.34731658829071227
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/original_unet_rgb/original_unet_rgb
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/original_unet_rgb/model_best.pth
Model and weights loaded..
---------------------------------
=============== original_unet_rgb VALIDATION ==============
[Validation] mIOU                : 0.5507454183944495
[Validation] Background          : 0.9843786843114818
[Validation] Fire extinguisher   : 0.4841213420016004
[Validation] Backpack            : 0.6414469854077307
[Validation] Hand drill          : 0.40645767634115487
[Validation] Rescue randy        : 0.23732240391027967
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/original_unet/original_unet
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/original_unet/model_best.pth
Model and weights loaded..
---------------------------------
=============== original_unet VALIDATION ==============
[Validation] mIOU                : 0.5295820838408124
[Validation] Background          : 0.9807792027773314
[Validation] Fire extinguisher   : 0.4193263760002434
[Validation] Backpack            : 0.5366088069317468
[Validation] Hand drill          : 0.3776432439890993
[Validation] Rescue randy        : 0.33355278950564116
=============================================================
Data Directory: /data/pst900_thermal_rgb/pstnet/data/data/PST900_RGBT_Dataset/test/
Weights Path: /data/pst900_thermal_rgb/pstnet/data//weights/fast_scnn_rgb
Model and weights loaded ..
============fast_scnn_rgb VALIDATION ===============
[Validation] mIOU                : 0.4838450698724359
[Validation] Background          : 0.9892051690384035
[Validation] Fire extinguisher   : 0.3466270630146492
[Validation] Backpack            : 0.6703110720105171
[Validation] Hand drill          : 0.20706366818253835
[Validation] Rescue randy        : 0.20601837711606769
=============================================================
Data Directory: /data/pst900_thermal_rgb/pstnet/data/data/PST900_RGBT_Dataset/test/
Weights Path: /data/pst900_thermal_rgb/pstnet/data//weights/fast_scnn
Model and weights loaded ..
============fast_scnn VALIDATION ===============
[Validation] mIOU                : 0.4695964649623636
[Validation] Background          : 0.9880904303550768
[Validation] Fire extinguisher   : 0.3619966572407812
[Validation] Backpack            : 0.637760257441924
[Validation] Hand drill          : 0.1549934449637338
[Validation] Rescue randy        : 0.2051415348103011
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/mfnet/mfnet
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/mfnet/model_best.pth
Model and weights loaded..
---------------------------------
=============== mfnet VALIDATION ==============
[Validation] mIOU                : 0.5534237277502692
[Validation] Background          : 0.9854967515526177
[Validation] Fire extinguisher   : 0.6361893921904445
[Validation] Backpack            : 0.6189536285488376
[Validation] Hand drill          : 0.36580730947087353
[Validation] Rescue randy        : 0.1606715569885718
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/rtfnet_50/rtfnet_50
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/rtfnet_50/model_best.pth
Model and weights loaded..
---------------------------------
=============== rtfnet_50 VALIDATION ==============
[Validation] mIOU                : 0.48719941952092977
[Validation] Background          : 0.9882008053172616
[Validation] Fire extinguisher   : 0.4532873567071127
[Validation] Backpack            : 0.694587505553467
[Validation] Hand drill          : 0.005090222930296148
[Validation] Rescue randy        : 0.2948312070965113
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/rtfnet/rtfnet
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/rtfnet/model_best.pth
Model and weights loaded..
---------------------------------
=============== rtfnet VALIDATION ==============
[Validation] mIOU                : 0.5527682214411972
[Validation] Background          : 0.9887507507473354
[Validation] Fire extinguisher   : 0.4549748988212398
[Validation] Backpack            : 0.7258058566938226
[Validation] Hand drill          : 0.2515920568135076
[Validation] Rescue randy        : 0.3427175441300804
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/pstnet/pstnet
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/pstnet/model_best.pth
Model and weights loaded..
---------------------------------
=============== pstnet VALIDATION ==============
[Validation] mIOU                : 0.6765786209506062
[Validation] Background          : 0.9883811448903448
[Validation] Fire extinguisher   : 0.6814184290314683
[Validation] Backpack            : 0.6990100748552718
[Validation] Hand drill          : 0.515131397225224
[Validation] Rescue randy        : 0.498952058750722
=============================================================
---------- DATA PATHS: ----------
Model File: /data/pst900_thermal_rgb/pstnet/data/architectures/pstnet_thermal/pstnet_thermal
Weight File: /data/pst900_thermal_rgb/pstnet/data/weights/pstnet_thermal/model_best.pth
Model and weights loaded..
---------------------------------
=============== pstnet_thermal VALIDATION ==============
[Validation] mIOU                : 0.6836562966338
[Validation] Background          : 0.9885413838836438
[Validation] Fire extinguisher   : 0.7012861940681571
[Validation] Backpack            : 0.6920069309924344
[Validation] Hand drill          : 0.5360640615493317
[Validation] Rescue randy        : 0.5003829126754333
=============================================================
```

## Repository Structure

```
.
|-- README.md
|-- code
|   |-- FastSCNN
|   |   |-- LICENSE
|   |   |-- README
|   |   |-- README.md
|   |   |-- data_loader
|   |   |   |-- __init__.py
|   |   |   |-- cityscapes.py
|   |   |   `-- penn_dataset.py
|   |   |-- demo.py
|   |   |-- eval.py
|   |   |-- license.txt
|   |   |-- models
|   |   |   |-- __init__.py
|   |   |   `-- fast_scnn.py
|   |   |-- png/
|   |   |-- train.py
|   |   `-- utils_fscnn
|   |       |-- __init__.py
|   |       |-- loss.py
|   |       |-- lr_scheduler.py
|   |       |-- metric.py
|   |       `-- visualize.py
|   |-- eval_iou.py
|   |-- generate_inference_table.py
|   |-- pst_data_definition.py
|   `-- pst_inference.py
|-- data
|   |-- architectures
|   |   |-- README
|   |   |-- erfnet
|   |   |   `-- erfnet.py
|   |   |-- erfnet_rgb
|   |   |   `-- erfnet_rgb.py
|   |   |-- fast_scnn
|   |   |   `-- fast_scnn.py
|   |   |-- fast_scnn_rgb
|   |   |   `-- fast_scnn_rgb.py
|   |   |-- mavnet
|   |   |   `-- mavnet.py
|   |   |-- mavnet_rgb
|   |   |   `-- mavnet_rgb.py
|   |   |-- mfnet
|   |   |   `-- mfnet.py
|   |   |-- original_unet
|   |   |   |-- original_unet.py
|   |   |   `-- unet_parts.py
|   |   |-- original_unet_rgb
|   |   |   |-- original_unet_rgb.py
|   |   |   `-- unet_parts.py
|   |   |-- pstnet
|   |   |   `-- pstnet.py
|   |   |-- pstnet_thermal
|   |   |   `-- pstnet_thermal.py
|   |   |-- resunet
|   |   |   |-- __init__.py
|   |   |   `-- resunet.py
|   |   |-- resunet_rgb
|   |   |   |-- __init__.py
|   |   |   `-- resunet_rgb.py
|   |   |-- rtfnet
|   |   |   `-- rtfnet.py
|   |   `-- rtfnet_50
|   |       `-- rtfnet_50.py
|   |-- data
|   |   |-- PST900_RGBT_Dataset
|   |   |   |-- test/ {test data}
|   |   |   `-- train/ {train data}
|   `-- weights
|       |-- README
|       |-- erfnet
|       |   `-- model_best.pth
|       |-- erfnet_rgb
|       |   `-- model_best.pth
|       |-- fast_scnn
|       |   `-- model_best.pth
|       |-- fast_scnn_rgb
|       |   |-- README
|       |   `-- model_best.pth
|       |-- mavnet
|       |   `-- model_best.pth
|       |-- mavnet_rgb
|       |   `-- model_best.pth
|       |-- mfnet
|       |   |-- mfnet
|       |   |   |-- 255.pth
|       |   |   `-- README
|       |   `-- model_best.pth
|       |-- original_unet
|       |   `-- model_best.pth
|       |-- original_unet_rgb
|       |   `-- model_best.pth
|       |-- pstnet
|       |   `-- model_best.pth
|       |-- pstnet_thermal
|       |   `-- model_best.pth
|       |-- rtfnet
|       |   |-- model_best.pth
|       |   `-- readme
|       `-- rtfnet_50
|           |-- model_best.pth
|           `-- rtfnet_50
|               `-- readme
```
