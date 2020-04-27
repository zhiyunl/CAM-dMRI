
# CAM-dMRI
This repository uses Class Activation Mapping on diffusion MRI data, implemented in PyTorch.

## Class Activation Mapping (CAM)
Paper and Archiecture: [Learning Deep Features for Discriminative Localization][1]

Paper Author Implementation: [metalbubble/CAM][2]


*Class activation maps could be used to intepret the prediction decision made by the CNN. The left image below shows the class activation map of top 5 predictions respectively, you can see that the CNN is triggered by different semantic regions of the image for different predictions. The right image below shows the CNN learns to localize the common visual patterns for the same object class.*

## Code Description
**Dependancies**: pytorch-gpu, dipy.

**Usage**: first, set parameters in 'utils.py' accordingly.

Then, run 

`python3 main.py`

**Network**: VGG, GoogLeNet, AlexNet, Inception V3

**Data**: HCP life span dataset.
- Download the zipped files and upzip them.
- Preprocess using 'data.py', to create slices.
- Divde the total dataset into train group and test group. As you do that, [images must be arranged in this way][4]:
  ```
  ../DATA/train/old/*.png
  ../DATA/test/young/*.png
  ```
  
**Checkpoint**
- Checkpoint will be created in the checkpoint folder every ten epoch.
- By setting `RESUME = #`, you can resume from `checkpoint/$ModelN$#.pt`.

  [1]: https://arxiv.org/abs/1512.04150
  [2]: https://github.com/metalbubble/CAM

