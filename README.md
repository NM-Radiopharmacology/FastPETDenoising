# FastPETDenoising
Adaptation of the U-Net architecture for denoising fast-acquisition PET images.

## CNN Architectures
The U-Net is a CNN architecture designed by Ronneberger, Fischer and Brox (2015), for the purpose of biomedical image segmentation. Here, an adaptation of this architecture was built for biomedical image denoising. In particular, this project was developed for restoring fast-acquisition PET images to their standard quality. 2.5D and 3D implementations are provided.

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
WIP...

### Inference
WIP...