# Penn Subterranean Thermal 900

Repository for **PST900 RGB-Thermal Calibration, Dataset and Segmentation Network** | C++, Python, PyTorch

If you have any questions regarding this dataset, please raise a GitHub issue here or reach out to us at sshreyas@seas.upenn.edu *(Shreyas S. Shivakumar)* or rodri651@seas.upenn.edu *(Neil Rodrigues)*.

[![PST900Video](http://img.youtube.com/vi/8nZ-uYN7BG0/0.jpg)](http://www.youtube.com/watch?v=8nZ-uYN7BG0)

If you use this dataset, please cite [our arXiv paper](https://arxiv.org/abs/1909.10980) below:
```
@misc{shivakumar2019pst900,
    title={PST900: RGB-Thermal Calibration, Dataset and Segmentation Network},
    author={Shreyas S. Shivakumar and Neil Rodrigues and Alex Zhou and Ian D. Miller and Vijay Kumar and Camillo J. Taylor},
    year={2019},
    eprint={1909.10980},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

If you wish to cite our overall DARPA Subterranean Challenge [autonomy system](https://arxiv.org/abs/1909.09662), please cite:
```
@misc{miller2019tunnel,
    title={Mine Tunnel Exploration using Multiple Quadrupedal Robots},
    author={Ian D. Miller and Fernando Cladera and Anthony Cowley and Shreyas S. Shivakumar and Elijah S. Lee and Laura Jarin-Lipschitz and Akhilesh Bhat and Neil Rodrigues and Alex Zhou and Avraham Cohen and Adarsh Kulkarni and James Laney and Camillo Jose Taylor and Vijay Kumar},
    year={2019},
    eprint={1909.09662},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

# Index

1. [Dataset](/README.md)
2. [RGB-T Calibration](/RGBTCalibration.md)
3. [Semantic Segmentation](/SemanticSegmentation.md)
4. [Hardware](/HardwareSpec.md)

# PST900 Dataset

| Dataset  | Total Samples | Train/Test Split | Download Link | Size | 
| ------------- | ------------- | ------------ | ------------ | ------------ | 
| PST900 RGB-T  | 894  | Yes | [Google Drive](https://drive.google.com/open?id=1hZeM-MvdUC_Btyok7mdF00RV-InbAadm) | 1.4GB | 
| PST-RGB   | 3359*  | No | [Google Drive](https://drive.google.com/file/d/1E455FCQ7CjE5VrYwr9msuNL8_5E5TTdn/view?usp=sharing) | 3.1GB | 

```
* our paper mentions 3416 annotated RGB images. Additional images will be provided on request.
```

## PST900 RGB-T

#### Sensor Head

![platform](/imgs/robotplatform.png)

#### Example Image

![rgbtdata](/imgs/pstrgbt.png)

The directory structure for this dataset is as follows:
```
.
├── test/                                           # Train data split (606 pairs)
|   ├── rgb/                                        # RGB Images 
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png   
|   |   ├── ...
|   ├── thermal/                                    # Thermal Images (FLIR's AGC 8-bit)
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png
|   |   ├── ...
|   ├── thermal_raw/                                # Thermal Images (RAW 16-bit)
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png
|   |   ├── ...
|   ├── depth/                                      # Depth image from stereo depth estimation
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png
|   |   ├── ...
|   ├── labels/                                     # Human annotated per-pixel label
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png
|   |   ├── ...
├── train/                                          # Test data split (288 pairs)
|   ├── rgb/                                        # RGB Images 
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png   
|   |   ├── ...
|   ├── thermal/                                    # Thermal Images (FLIR's AGC 8-bit)
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png
|   |   ├── ...
|   ├── thermal_raw/                                # Thermal Images (RAW 16-bit)
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png
|   |   ├── ...
|   ├── depth/                                      # Depth image from stereo depth estimation
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png
|   |   ├── ...
|   ├── labels/                                     # Human annotated per-pixel label
|   |   ├── 31_bag1a_rect_rgb_frame0000000007.png
|   |   ├── ...
```

## PSTRGB

#### Example Images

![rgbdata](/imgs/pstrgb.png)

The directory structure for this dataset is as follows:
```
├── rgb/                                        # RGB Images 
|   ├── 01_levine_rgb_1_rdb_bag_100109   
|   ├── ...
├── labels/                                     # Human annotated per-pixel label
|   ├── 01_levine_rgb_1_rdb_bag_100109
|   ├── ...
```


## Toolkit

We provide a basic set of tools for working with this dataset. This tools are mainly focused on how to read and represent the RGB, Thermal, Thermal (16-bit) and Labels.

Basic utilities can be found in the **toolkit/utilities.py** script.

### Basic Utilities

A demo of the utilities are provided in **utilities.py**; this can be run as a python script. Provide the location to the dataset to the *pst900_path* variable and run the following command:

```
python3 utilities.py
```

#### 1. Data loader

Once you select a split type (either Train or Test, or another custom split), you can use our dataset sample fetch script:

```
# Example dataset sample loader
rgb, depth, thermal, thermal_raw, label = utils.get_sample(102)
```

and each modality should look like this:
```
==== Loading sample ====
Loaded RGB image | H: 720, W: 1280, C: 3, Type: uint8
Loaded Depth image | H: 720, W: 1280, C: 1, Type: uint16
Loaded Thermal image | H: 720, W: 1280, C: 1, Type: uint8
Loaded Thermal Raw image | H: 720, W: 1280, C: 1, Type: uint16
Loaded label image | H: 720, W: 1280, C: 1, Type: uint8
```

**Note**: Depth and Thermal Raw (16 bit) must be loaded as 16-bit images.

The RGB and Thermal (8-bit) can be easily viewed and should look like this:

![RGB](/imgs/utils_rgb_loader.png)
![Thermal](/imgs/utils_thermal_loader.png)

#### 2. Thermal Hole Filling

The following method will perform a simple hole filling in the thermal images. The holes are a result of projecting the information from the Thermal camera onto the RGB camera. The holes are a result of parallax, resulting in a many-to-one mapping of pixels in RGB into the Thermal camera plane.

```
# Example hole filling for Thermal image
thermal_filled = utils.fill_thermal(thermal)
```

The output should look like this:

![hole_filled](/imgs/thermal_hole_filling.png)

#### 3. Label visualization

The labels are stored as uint8 images, with each pixel representing the class index {0..4}. This class indices can be mapped to colors for easier visualization as follows:

```
# Example colormapping for Label image
colormapped_label = utils.visualize_label(label)
```

The resulting image should look like this:

![label_vis](/imgs/utils_label_vis.png)
