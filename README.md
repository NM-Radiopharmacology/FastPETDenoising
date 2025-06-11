# FastPETDenoising
Adaptation of the U-Net architecture for denoising fast-acquisition PET images.

## CNN Architectures
The U-Net is a CNN architecture designed by Ronneberger, Fischer and Brox (2015), for the purpose of biomedical image segmentation. Here, an adaptation of this architecture was built for biomedical image denoising. In particular, this project was developed for restoring fast-acquisition PET images to their standard quality. 2.5D and 3D implementations are provided.

### 3D U-Net
The network's input is, by default, a 128×128×128 volume. The number of encoder/decoder blocks was reduced to three to reduce computational complexity, as we are dealing with 3D images. Each encoder block is composed by two convolutional layers followed by a downsampling layer. Each convolutional layer contains a **convolution** that doubles the number of feature maps (except in the input block, where, by default, 64 feature maps are extracted from the 1-channel input) and a **ReLU activation**. Batch normalisation was removed, as the scale of each PET image, in particular, is not fixed-range and contains important physiological information which musn't be altered. Downsampling is done through **average pooling**, instead of max pooling, as denoising is not a classification problem.

<img src="/figures/unet_3D.png" alt="3D U-Net architecture" style="max-width: 75%; height: auto;">

The implementation of this CNN can be found in `networks/unet_three_d.py`.

### 3-channel 2.5D U-Net
This "conventional" 2.5D approach was implemented for axial-based denoising of the PET images. The network's input is, by default, a 3-channel 144×144 image. The 3 channels correspond to each axial slice (middle channel) and its immediately prior and successful slices. The network's first layer flattens the 3 channels into one, and a U-Net architecture follows, contemplating the same denoising-specific alterations described in the 3D U-Net section above.

<img src="/figures/unet_3channel-2.5D.png" alt="3-channel 2.5D U-Net architecture" style="max-width: 95%; height: auto;">

The implementation of this CNN can be found in `networks/unet_two_and_a_half_d.py`.

### 1-channel 2.5D U-Net

This 1-channel 2.5D approach was implemented for 2D denoising of all three anatomical planes - axial, coronal and sagittal. The network's input is, by default, a 144×144 image patch. U-Net architecture contemplates the same denoising-specific alterations described in the 3D U-Net section above.

<img src="/figures/unet_1channel-2.5D.png" alt="1-channel 2.5D U-Net architecture" style="max-width: 95%; height: auto;">

The implementation of this CNN can be found in `networks/unet_two_and_a_half_d.py`.

## Installation & Usage
P.S.: Using a virtual environment is recommended! (Git Bash: `source <path-to-venv>/Scripts/activate` - Windows)

**PyTorch must be installed beforehand!!** [Refer to their website and install PyTorch](https://pytorch.org/get-started/locally/) with support for your hardware ([CUDA](https://developer.nvidia.com/cuda-toolkit), CPU).

Only then (on Git Bash):

```
git clone https://github.com/NM-Radiopharmacology/FastPETDenoising.git
cd FastPETDenoising
pip install -r requirements.txt
```

### Training

#### Step 1: Dataset & training parameters ⟶ `1_define_training_configuration.py`
The pipeline for all networks expects 3D input images - the network-specific inputs are managed when loading data before training. The user need only provide the paths to the training images and to the respective reference images. This is done by running `1_define_training_configuration.py`. During this step, the user can also alter some training parameters, like learning rate, optimizer, loss function, etc.

**Important!** Training images and respective references must be ordered alphabetically! For instance:

```
training_images/ 
    ├── 001_fastPET.nii.gz 
    ├── 002_fastPET.nii.gz
    ├── 003_fastPET.nii.gz
    └── ...
reference_images/ 
    ├── 001_fastPET_ref.nii.gz 
    ├── 002_fastPET_ref.nii.gz
    ├── 003_fastPET_ref.nii.gz
    └── ...
```

A `configuration.json` file is created with the training configuration and dataset paths.

#### Step 2: Training ⟶ `2_train.py`

After generating the `configuration.py` file, the user must run `2_train.py`. This script will automatically find the configuration file in the repository's directory and start a training instance - `training_YYYY-MM-DD_HH-MM`. This folder will store the configuration file, all models, training logs and loss plots.

#### Step 3: Plotting training progress (optional) ⟶ `3_loss_plot.py`

Running `3_loss_plot.py` will generate a `matplotlib.Figure` pop-up window which plots training and validation losses live.

### Inference

#### Step 4: Applying the model ⟶ `4_inference.py`

To apply the model, the user need only run `4_inference.py` and introduce path to the images to denoise. This script will find the local training instances, request which one to employ and, subsequently, which model to use. From `coniguration.json`, relevant model parameters (e.g. CNN architecture) are loaded automatically.

### Pre-trained models

Pre-trained models will be released soon...
