"""
script to apply the model
"""
import os
import shutil
import sys
import itk
import numpy as np
import torch
import json
from networks.unet_three_d import UNet3D
from networks.unet_two_and_a_half_d import UNet25D
from utilities import printdt, pad_to_patch_size, denoise_volume_gaussian, denoise_slice_gaussian
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

training_instances = ['pretrained_models/' + item for item in os.listdir(os.path.join(os.getcwd(), 'pretrained_models'))
                      if item.startswith('training')]
if os.path.exists('pretrained_models/unet_1channel-2.5D'):
    training_instances.insert(0, 'pretrained_models/unet_1channel-2.5D')
if os.path.exists('pretrained_models/unet_3channel-2.5D'):
    training_instances.insert(0, 'pretrained_models/unet_3channel-2.5D')
if os.path.exists('pretrained_models/unet_3D'):
    training_instances.insert(0, 'pretrained_models/unet_3D')
if len(training_instances) == 0:
    print('No training instances found!')
    sys.exit()

print("The following training instances were found:")
for i, instance in enumerate(training_instances):
    print(f"{i}: {instance}")

training_folder = None
all_models = []
while model_path is None:
    i = input("Introduce which training instance to use (corresponding index): ")
    try:
        training_folder = training_instances[int(i)]
        model_path = os.path.join(os.getcwd(), training_folder)
        all_models = [item for item in os.listdir(model_path) if item.endswith('.pt')]
        if len(all_models) == 0:
            print("No models found!")
            model_path = None
    except (IndexError, ValueError):
        continue

use_last_model = True
folds = [mdl.split('_')[-2] for mdl in all_models]
folds = list(set(folds))
models = []
for fld in folds:
    current_fold_models = []
    for mdl in all_models:
        if fld in mdl:
            current_fold_models.append(mdl)
    current_fold_models.sort()
    if use_last_model:
        models.append(current_fold_models[-1])
    else:
        models.append(current_fold_models[0])
models.sort()

with open(os.path.join(training_folder, 'configuration.json')) as json_file:
    configuration = json.load(json_file)

patch_size = configuration['patch_size']
network_configuration = configuration['network_configuration']
pooling = configuration['pooling']

out_path = os.path.join(out_path, f'denoised_{network_configuration}')
if not os.path.exists(out_path):
    os.makedirs(out_path)
else:
    shutil.rmtree(out_path)
    os.makedirs(out_path)

printdt(f"starting inference from all available models in {training_folder} ({len(models)})")
for img in tqdm(os.listdir(img_path), file=sys.stdout, desc=datetime.now().strftime("%d/%m/%Y %H:%M:%S")):
    arr = np.asarray(itk.imread(os.path.join(img_path, img)))  # Array of the 3D image to split in patches

    original_shape = None
    if arr.shape[0] < patch_size[0] or arr.shape[1] < patch_size[1] or arr.shape[2] < patch_size[2]:
        original_shape = arr.shape
        arr = pad_to_patch_size(arr, patch_size)
    meta = dict(itk.imread(os.path.join(img_path, img)))

    ensemble_arr = np.zeros_like(arr)
    ensemble_counter = 0
    for model_path in models:

        model = None
        if network_configuration == "3D":
            model = UNet3D(pooling=pooling)
        elif network_configuration == "1channel-2.5D":
            model = UNet25D(in_channels=1, pooling=pooling)
        elif network_configuration == "3channel-2.5D":
            model = UNet25D(in_channels=3, pooling=pooling)

        if model is not None:
            model.to(device)
            model.load_state_dict(torch.load(os.path.join(training_folder, model_path),
                                             weights_only=True, map_location=torch.device(device)))

            den_arr = None
            if network_configuration == "3D":
                den_arr = denoise_volume_gaussian(arr, model, patch_size=patch_size, device=device)
            elif network_configuration == "1channel-2.5D":
                depth = arr.shape[0]  # z; axial
                height = arr.shape[1]  # y; coronal
                width = arr.shape[2]  # x; sagittal

                den_arr = np.zeros(arr.shape)
                for i in range(depth):  # axial denoising
                    den_slice = denoise_slice_gaussian(arr[i, :, :], model, patch_size=patch_size)
                    den_arr[i, :, :] = den_arr[i, :, :] + den_slice
                for j in range(height):  # coronal denoising
                    den_slice = denoise_slice_gaussian(arr[:, j, :], model, patch_size=patch_size)
                    den_arr[:, j, :] = den_arr[:, j, :] + den_slice
                for k in range(width):  # sagittal denoising
                    den_slice = denoise_slice_gaussian(arr[:, :, k], model, patch_size=patch_size)
                    den_arr[:, :, k] = den_arr[:, :, k] + den_slice
                den_arr = den_arr / 3  # average of axial, coronal and sagittal-based denoising
            elif network_configuration == "3channel-2.5D":
                den_arr = np.zeros(arr.shape)
                for i in range(arr.shape[0]):  # axial denoising
                    if i == 0 or i == arr.shape[0] - 1:
                        if i == 0:
                            slice_to_reflect = arr[i + 1, :, :]
                        if i == arr.shape[0] - 1:
                            slice_to_reflect = arr[arr.shape[0] - 2, :, :]

                        slice_to_denoise = np.zeros([3, arr.shape[1], arr.shape[2]])
                        slice_to_denoise[0, :, :] = slice_to_reflect
                        slice_to_denoise[1, :, :] = arr[i, :, :]
                        slice_to_denoise[2, :, :] = slice_to_reflect
                    else:
                        slice_to_denoise = arr[i - 1:i + 2, :, :]

                    den_slice = denoise_slice_gaussian(slice_to_denoise, model, patch_size=patch_size)
                    den_arr[i, :, :] = den_arr[i, :, :] + den_slice

            if den_arr is not None:
                ensemble_counter = ensemble_counter + 1
                ensemble_arr = ensemble_arr + den_arr

    # N-fold ensembling
    ensemble_arr = ensemble_arr / ensemble_counter
    if original_shape is not None:
        croppings = tuple(slice(0, dim) for dim in original_shape)
        ensemble_arr = ensemble_arr[croppings]
    image_itk = itk.image_from_array(ensemble_arr)
    for k, v in meta.items():
        image_itk[k] = v

    for extension in ['.nii.gz', '.nii', '.nrrd']:
        if img.endswith(extension):
            img = img.replace(extension, f"_{network_configuration}{extension}")
            break

    itk.imwrite(image_itk, os.path.join(out_path, img))
