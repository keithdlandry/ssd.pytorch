# SSD Single Shot Detector for Small Basketball Detection
A modified [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325)
from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.
  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd). 


## Introduction

The basis of this project comes an original [implementation](https://github.com/amdegroot/ssd.pytorch) 
of the SSD described in the [original](http://arxiv.org/abs/1512.02325) paper. See README.md for information 
regarding this, general requirements, and installation instructions. One of the limitations of the SSD 
is the detection of small objects. Since our basketballs are as small as 20 pixels across, some changes needed to me made.


## Base Network

The base network of the altered SSD is still [VGG-16](https://arxiv.org/abs/1409.1556). 
Pretrained weights for this network can be downloaded as followed. 

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

## Altered SSDs
I have created 3 different SSD architectures to detect small basketballs. All architectures have the
option of only allowing square prior boxes. This is recommended. There is also an option for detecting
people as well as balls. This is turned off by default.

#### Basic SSD for Large images
The original SSD was designed to work on 300 x 300 images. This had to be changed if there was any chance 
at detecting small objects. The architecture was altered to allow input images of 1166 x 1166. Other than 
allowing for larger input size, nothing else was changed.
* Network name '300' (should really be changed to '1166')

#### Truncated SSD
The single shot detector detects objects at different scales by classifying and localizing these objects 
from the output of different layers of the network. The smallest objects coming from earlier layers 
and large object coming from the later layers. Since we desire this network to specialize detect small 
objects, the later layers are not usefull. This network is truncated after the first detection layer 
associated with the smallest objects. This allows for increased speed. Unfortunately the pretrained 
weights of the VGG-16 backbone can not be used as is. This network is also designed for 1166 x 1166 
input images.
* Network name 'trunc'

#### Lookahead SSD
The "lookahead" is an option for the basic SSD for large images which passes information back one layer
to the first detection layer. This allows for the detection layer to have some sort of context from 
which to make descisions. This seems to allow for slightly better performance but at a cost of longer 
compute time.


## Usage

#### Network configuration
In the original project, multiple changes in many different files were required to configure the network.
This has been reworked to be handled in one place, master_config.py. Here the number of channels and 
type of layers of the base network are defined as well as the extra layers added after the base VGG-16
backbone. 

Prior bounding box configuration is also set here. The size of the feature maps going into the detection 
and localization layers must be specified. The size of the bounding boxes for each detection layer are 
stated. Changing these may allow for different sized objects to be detected. I believe the default sizes 
are best since they correspond roughly to the receptive fields of the detection layers.

#### Training
Currently training is done in two files, one for regular networks and the other for the lookahead option. 
They are essentially identical and should be converted to a single file with the lookahead option as 
I've done with testing.

The default parameters of learning rate (1e-5), momentum (0.9), weight decay (5e-4), and gamma (0.1)
worked very well for images from the initial Beverly Hills Jewish Center recording.

Training images are specified by the --id_file flag. This file has a list of image ids. For example:
```text
00001
00002
00003
```
Trained weights are saved after every 1000 iterations as well as when the max iteration is reached.

#### Testing
Testing is handled in test_general_net_arch.py. Network configuration is set by the --net_name and --lookahead 
flags. The net name must be one of the keys in the configs dictionary in master_config.py. The lookahead 
flag is a boolean determining weather or not to let the information flow backwards to the first detection 
layer.

Images are identified in one of two ways. The first is with a text file specified with the --id_file 
flag containing a list of image ids as with training.

The second way is using the --id_start and --id_end flags. These take integers (not including zeros). 
The --id_zeropadding flag is also required using this method to allow for the correct number of zeros
to be added to the image id.

In both methods, the --file_prefix flag is required to be everything in the image file name prior to 
the image id. The --file_type is also required to set the image type (.png, .jpg, etc.). Image directory 
must also be specified with the --im_dir flag.

The output from testing is a json file with the image id, bounding boxes, and confidences for each detection. 
This json file can be used in create_gif.py to draw bounding boxes on the images, which can then be 
converted to an animated gif using ffmpeg.

## Dataset
The only dataset used for training so far is the initial Beverly Hills Jewish Center recording. A data 
loader designed to load images from this dataset is found in data/bhjc20180123_bball/bhjc.py. An annotation 
transformer for the dataset is also found there. These should also work on similar datasets with the 
annotation format. The only specifications that need to be altered are the image and annotation file 
prefix. There is an option to pull images and annotations from the genius sports s3 bucket but this 
is slower and must be set by hand in bhjc.py.

Durring training the loaded images are passed through transformations which alter the colors, crop, 
zoom, etc. the images. This should allow for better generalization as well as provide a larger training
dataset since the same image can be used multiple times with different transformations.

## Example

![alt text](https://gitlab.betgenius.net/computer_vision/basketball_detection/blob/ball-only/data/output_imgs/unannot/lookahead1166_300_vanilla_thresh.3/leftcam_detect_unannot_00803.png "detection example")
