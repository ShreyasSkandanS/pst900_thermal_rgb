# Penn Subterranean Thermal 900

Repository for **PST900 RGB-Thermal Calibration, Dataset and Segmentation Network** | C++, Python, PyTorch


# PST900 Dataset

| Dataset  | Total Samples | Train/Test Split | Download Link | Size | 
| ------------- | ------------- | ------------ | ------------ | ------------ | 
| PST900 RGB-T  | 894  | Yes | [Google Drive](https://drive.google.com/file/d/1f4QT7RFfHIEMYW80jrbBs3qo5qjKou9M/view?usp=sharing) | 1.4GB | 
| PST-RGB   | 3359*  | No | [Google Drive](https://drive.google.com/file/d/1E455FCQ7CjE5VrYwr9msuNL8_5E5TTdn/view?usp=sharing) | 3.1GB | 

```
* our paper mentions 3416 annotated RGB images. The additional images can be provided on request.
```

## PST900 RGB-T

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
