# Hardware

## Sensor Head Specification

For our dataset, we use the following:

### 1. [Stereolabs Zed Mini](https://www.stereolabs.com/zed-mini/): 

Example Zed Mini image stream from Tunnel Circuit:

[![StereolabsZedMini](https://img.youtube.com/vi/mCiGOzolTfE/0.jpg)](https://www.youtube.com/watch?v=mCiGOzolTfE)

* Multiple different depth camera systems were tested before I decided that the Zed Mini was the best option for our requirements. 

* The depth estimation backbone is passive stereo matching. This seems like an odd choice of camera to send into a dark cave/mine without natural illumination. However, with our onboard active illumintation system we were able to achieve good depth estimation performance underground.

* Since this is the primary sensor for artifact recognition, one of the primary requirements was having good quality RGB imagery, and in my comparisons with the ZED's RealSense counterparts, I believe the quality of the RGB cameras on the Zed outperform it's alternatives.

* The Zed mini is also able to better handle dust, which was an important factor in our decision. Active illumination in a dust-filled environment brings interesting challenges with visual degradation of imagery. I am very happy with the performance of this sensor in these environments. I suspect that the choice of sensor along with the fact that this is a rolling shutter system, makes individual dust particles less prominent.

* Lastly, the icing on the cake - the stereo camera calibration tool. Having a tool this easy to use in the field is invaluable. Hats off to the Stereolabs team for building such an intuitive calibration software. As someone who has spent the last 3 years calibrating stereo cameras, you have my respect and gratitude.

### 2. [FLIR Boson 320](https://www.flir.com/products/boson/):

![RGBImage](/imgs/clr_rgb.png)
![Thermal](/imgs/img_thermal.png)

* This camera appears to be somewhat of a favourite among the SubT teams. I noticed them on numerous platforms at Tunnel Circuit and it isn't surprising. I have used the Boson 320 in the 50deg and 92deg variants and they are excellent devices that require very little in the way of compute or set-up to get running.

* For this dataset, we use the FLIR Boson 320 2.3mm 92 degree HFOV model. We use the 60Hz variant and record our data in both 8-bit AGC as well as raw 16-bit. 

* The 320 variant seemed sufficient for our task; the 640 must be much nicer but it is also significantly more expensive. 

### 3. [Stratus Lighting](https://www.stratusleds.com/aerial-leds)

![Lighting](/imgs/lighting.gif)

* There's not much to say about these lights. They're lights. They're bright lights. They're very bright lights.


