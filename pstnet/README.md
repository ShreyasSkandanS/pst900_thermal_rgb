# PST900 Inference & Verification

## Generating Table II

You can generate all the results from Table II of our paper by running the following script:

```
python3 generate_inference_table.py
```

Before you do that, make sure you have done the following:

* Downloaded the dataset (link in data/data folder)
* Downloaded the pre-trained models (link in data/weights folder)
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

Make changes to the following lines:
```
# Directory of model files
ARTIFACT_DETECTION_DIR = "/data/pst900_thermal_rgb/pstnet/data/"
# Directory which contains batch of input images
ARGS_INFERENCE_DIR = ARTIFACT_DETECTION_DIR + "data/PST900_RGBT_Dataset/test/"
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
