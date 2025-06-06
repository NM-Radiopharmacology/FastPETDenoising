"""
script to apply the model
"""
import os
import sys
import itk
import numpy as np
import torch
import json
from networks.unet_three_d import UNet3D
from networks.unet_two_and_a_half_d import UNet25D
from utilities import printdt, denoise_volume_gaussian, denoise_slice_gaussian
from tqdm import tqdm
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = None
out_path = None
network_configuration = None
model_path = None

while img_path is None:
    img_path = input("Enter path to the images to denoise: ")

while out_path is None:
    out_path = input("Enter path in which to save the folder with the denoised images: ")
os.makedirs(os.path.join(out_path, 'denoised'))
out_path = os.path.join(out_path, 'denoised')

training_instances = [item for item in os.listdir(os.getcwd()) if item.startswith('training')]
if len(training_instances) == 0:
    print('No training instances found!')
    sys.exit()

print("The following training instances were found:")
for i, instance in enumerate(training_instances):
    print(f"{i}: {instance}")

training_folder = None
while model_path is None:
    i = input("Introduce which training instance to use (corresponding index): ")
    try:
        training_folder = training_instances[int(i)]
        model_path = os.path.join(os.getcwd(), training_folder)
        models = [item for item in os.listdir(model_path) if item.endswith('.pt')]
        if len(models) == 0:
            model_folder = None
        else:
            model_path = os.path.join(model_path, models[-1])
    except (IndexError, ValueError):
        continue

with open(os.path.join(training_folder, 'configuration.json')) as json_file:
    configuration = json.load(json_file)

patch_size = configuration['patch_size']
network_configuration = configuration['network_configuration']

model = None
if network_configuration == "3D":
    model = UNet3D()
elif network_configuration == "2.5D-1channel":
    model = UNet25D(in_channels=1)
elif network_configuration == "2.5D-3channel":
    model = UNet25D(in_channels=3)

if model is not None:
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path)))

    printdt(f"starting inference from model {training_folder}/{model_path.split(os.sep)[-1]}")
    for img in tqdm(os.listdir(img_path), file=sys.stdout, desc=datetime.now().strftime("%d/%m/%Y %H:%M:%S")):
        arr = np.asarray(itk.imread(os.path.join(img_path, img)))  # Array of the 3D image to split in patches
        meta = dict(itk.imread(os.path.join(img_path, img)))

        if network_configuration == "3D":
            den_arr = denoise_volume_gaussian(arr, model, patch_size=patch_size, device=device)
        else:
            depth = arr.shape[0]  # z; axial
            height = arr.shape[1]  # y; coronal
            width = arr.shape[2]  # x; sagittal

            den_arr_z = np.zeros(arr.shape)
            den_arr_y = np.zeros(arr.shape)
            den_arr_x = np.zeros(arr.shape)

            for z in range(depth):
                im2d = arr[z, :, :]
                den_im = denoise_slice_gaussian(im2d, model)
                den_arr_z[z, :, :] = den_im

            for y in range(height):
                im2d = arr[:, y, :]
                den_im = denoise_slice_gaussian(im2d, model)
                den_arr_y[:, y, :] = den_im

            for x in range(width):
                im2d = arr[:, :, x]
                den_im = denoise_slice_gaussian(im2d, model)
                den_arr_x[:, :, x] = den_im

            den_arr = (den_arr_x + den_arr_y + den_arr_z) / 3

        image_itk = itk.image_from_array(den_arr)
        for k, v in meta.items():
            image_itk[k] = v

        for extension in ['.nii.gz', '.nii', '.nrrd']:
            if img.endswith(extension):
                img = img.replace(extension, f"_unet3d{extension}")
                break

        itk.imwrite(image_itk, os.path.join(out_path, img))
