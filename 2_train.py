"""
script to train the network
training is performed from the configuration.json file in the project's directory
run 1_define_training_configuration.py to configure training and create configuration.json accordingly
"""
import os
import numpy as np
import pandas as pd
from utilities import printdt, denoise_volume_gaussian, denoise_slice_gaussian, denoise_patch
from make_torch_dataset import MakeTorchDataset
from custom_losses import L1NormalizedLoss, CrossCorrLoss
from torch import nn
import json
import datetime
from tqdm import tqdm
import itk
import sys
from skimage.metrics import mean_squared_error
from networks.unet_three_d import UNet3D
from networks.unet_two_and_a_half_d import UNet25D
from torchsummary import summary
from torch.utils.data import DataLoader, Subset
import shutil
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
printdt(f"Is cuda available? {torch.cuda.is_available()} --> device: {device}")


# --------------------------------------------------------------------------------------------------- training utilities
loss_function = {
    "L1N": L1NormalizedLoss(),
    "MSE": nn.MSELoss(),
    "L1": nn.L1Loss(),
    "CC": CrossCorrLoss()
}


def trainer(data_loader, model, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(data_loader, file=sys.stdout,
                                desc=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")):

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = loss_function(out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validator(data_loader, model, loss_function, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for images, targets in tqdm(data_loader, file=sys.stdout,
                                    desc=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")):
            images = images.to(device)
            targets = targets.to(device)

            out = []

            for im in images:
                arr = im.detach().cpu().numpy()
                if network_configuration != "3channel-2.5D":
                    arr = arr.squeeze(0)
                arr = np.asarray(arr)

                if list(arr.shape) != list(patch_size):

                    if network_configuration == "3D":
                        den = denoise_volume_gaussian(arr, model, patch_size=patch_size)
                    elif network_configuration == "1channel-2.5D":
                        den = np.zeros(arr.shape)
                        for i in range(arr.shape[0]):   # axial denoising
                            den_slice = denoise_slice_gaussian(arr[i, :, :], model, patch_size=patch_size)
                            den[i, :, :] = den[i, :, :] + den_slice
                        for j in range(arr.shape[1]):   # coronal denoising
                            den_slice = denoise_slice_gaussian(arr[:, j, :], model, patch_size=patch_size)
                            den[:, j, :] = den[:, j, :] + den_slice
                        for k in range(arr.shape[2]):   # sagittal denoising
                            den_slice = denoise_slice_gaussian(arr[:, :, k], model, patch_size=patch_size)
                            den[:, :, k] = den[:, :, k] + den_slice
                        den = den / 3   # average of axial, coronal and sagittal-based denoising
                    elif network_configuration == "3channel-2.5D":
                        den = np.zeros(arr.shape)
                        for i in range(arr.shape[0]):   # axial denoising
                            if i == 0 or i == arr.shape[0] - 1:
                                if i == 0:
                                    slice_to_reflect = arr[i+1, :, :]
                                if i == arr.shape[0] - 1:
                                    slice_to_reflect = arr[arr.shape[0] - 2, :, :]

                                slice_to_denoise = np.zeros([3, arr.shape[1], arr.shape[2]])
                                slice_to_denoise[0, :, :] = slice_to_reflect
                                slice_to_denoise[1, :, :] = arr[i, :, :]
                                slice_to_denoise[2, :, :] = slice_to_reflect
                            else:
                                slice_to_denoise = arr[i-1:i+1, :, :]

                            den_slice = denoise_slice_gaussian(slice_to_denoise, model, patch_size=patch_size)
                            den[i, :, :] = den[i, :, :] + den_slice

                    den = np.expand_dims(den, axis=-1)
                    out.append(np.transpose(den, (3, 0, 1, 2)).astype(float))

                else:

                    den = denoise_patch(arr, model)     # directly applies model to full image
                    den = np.expand_dims(den, axis=-1)
                    out.append(np.transpose(den, (3, 0, 1, 2)).astype(float))

            out = np.asarray(out)
            out = torch.Tensor(out)
            out = out.to(device)
            loss = loss_function(out, targets)

            total_loss += loss.item()

    return total_loss / len(data_loader)


# ------------------------------------------------------------------------------------- training parameter configuration
if os.path.isfile('configuration.json'):
    with open('configuration.json') as json_file:
        configuration = json.load(json_file)
else:
    print('No configuration file found! Please run 1_define_training_configuration.py first!')
    sys.exit()

training_pairs = configuration["dataset"]["training_pairs"]
validation_folds = configuration["dataset"]["validation_folds"]

validation_image_size = configuration["dataset"]["validation_image_size"]

patch_size = configuration["patch_size"]

network_configuration = configuration['network_configuration']
loss_function_key = configuration['loss_function']
training_set_fraction = configuration['training_set_fraction']
perform_data_augmentation = configuration['perform_data_augmentation']

printdt("training configuration loaded")

# ------------------------------------------------------------------------------------------------------------- training
folder = f"training_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
os.makedirs(folder)
os.rename("configuration.json", os.path.join(folder, "configuration.json"))

for fold in validation_folds.keys():
    printdt(f"starting {fold}...")

    best_epoch = 0  # initialising epoch
    lr = configuration["learning_rate"]

    training_log_file = os.path.join(folder, f"training_log_{fold}.txt")
    open(training_log_file, "w").close()

    best_model = f"unet_{configuration['network_configuration']}_{fold}_{str(best_epoch).zfill(4)}.pt"
    last_model = best_model  # initialising model

    validation_lst = validation_folds[fold]
    if network_configuration == "3D":
        training_lst = [pair for pair in training_pairs if pair not in validation_folds[fold]]
    else:
        training_lst = []
        for pair in training_pairs:
            if pair not in validation_folds[fold]:
                img_shape = np.asarray(itk.imread(pair[0])).shape
                if network_configuration == "1channel-2.5D":
                    for n_dim, dim in enumerate(img_shape):
                        if n_dim == 2:
                            anatomical_plane = 'sagittal'   # x
                        elif n_dim == 1:
                            anatomical_plane = 'coronal'    # y
                        else:
                            anatomical_plane = 'axial'      # z
                        for i in range(dim):    # maybe remove 5% of slices in extremities?
                            training_lst.append([pair[0], pair[1], anatomical_plane, i])
                elif network_configuration == "3channel-2.5D":
                    anatomical_plane = 'axial'      # z
                    for i in range(img_shape[0]):
                        if i != 0 and i != img_shape[0] - 1:    # excluding first and last slices
                            training_lst.append([pair[0], pair[1], anatomical_plane, i])

    printdt(f'training set size: {len(training_lst)}')
    printdt(f'validation set size: {len(validation_lst)}')

    if network_configuration == "3D":
        columns = ['image', 'target']
    else:
        columns = ['image', 'target', 'anatomical_plane', 'coordinate']
    train_df = pd.DataFrame(training_lst, columns=columns)
    training_set = MakeTorchDataset(train_df, patch_size=patch_size, augmentations=perform_data_augmentation,
                                    network_configuration=network_configuration)
    valid_df = pd.DataFrame(validation_lst, columns=['images', 'targets'])

    printdt("batching validation set")
    validation_set = MakeTorchDataset(valid_df, augmentations=False, patch_size=patch_size,
                                      network_configuration=network_configuration,
                                      validation=True, validation_image_size=validation_image_size)
    validation_loader = DataLoader(validation_set, batch_size=configuration['batch_size'])

    printdt(f"calculating baseline loss ({loss_function_key})")

    valid_mse = []
    for valid_paths in tqdm(validation_lst, file=sys.stdout,
                            desc=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")):
        valid = np.asarray(itk.imread(valid_paths[0]))
        ref = np.asarray(itk.imread(valid_paths[1]))
        valid_mse.append(mean_squared_error(valid, ref))

    baseline_valid_loss = np.mean(valid_mse)

    with open(training_log_file, "a+") as f:
        f.write(f"epoch,training_loss({loss_function_key}),validation_loss({loss_function_key})\n")
        f.write(f"{best_epoch},,{baseline_valid_loss}\n")

    printdt("baseline loss:", baseline_valid_loss)

    printdt("loading model")

    model = None
    if network_configuration == "3D":
        model = UNet3D()
        input_size = (1, patch_size[0], patch_size[1], patch_size[2])
    elif network_configuration == "1channel-2.5D":
        model = UNet25D(in_channels=1)
        input_size = (1, patch_size[0], patch_size[1])
    elif network_configuration == "3channel-2.5D":
        model = UNet25D(in_channels=3)
        input_size = (3, patch_size[0], patch_size[1])
    else:
        printdt(f"unknown network_configuration: {network_configuration}")
        sys.exit()

    model.to(device)
    summary(model, input_size)

    printdt("starting training")
    best_valid_loss = baseline_valid_loss
    valid_loss = baseline_valid_loss
    for i in range(1, configuration["N_epochs"] + 1):
        indices = np.random.choice(len(train_df), size=int(len(train_df) * training_set_fraction), replace=False)
        training_subset = Subset(training_set, indices=indices)
        training_loader = DataLoader(training_subset, batch_size=configuration["batch_size"], shuffle=True)

        if i % configuration["learning_rate_decay_steps"] == 0:
            lr = lr * configuration["learning_rate_decay_rate"]

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)     # initialising optimiser

        printdt(f'[{i}] learning rate: {np.round(lr, 8)}')

        train_loss = trainer(training_loader, model, optimiser, loss_function[loss_function_key], device)

        if i % configuration["validation_every_N_epochs"] == 0 or i == 1:

            valid_loss = validator(validation_loader, model, loss_function[loss_function_key], device)

            printdt(f"[epoch {i}] training loss: {train_loss} | validation loss: {valid_loss}")

            with open(training_log_file, "a") as f:
                f.write(f'{i},{train_loss},{valid_loss}\n')

        else:
            printdt(f"[epoch {i}] training loss: {train_loss}")

            with open(training_log_file, "a") as f:
                f.write(f'{i},{train_loss}\n')

        if last_model != best_model:    # checking if the last model does not correspond to the current validation
            # minimum, as not to remove it in that case
            if os.path.isfile(os.path.join(folder, last_model)):
                os.remove(os.path.join(folder, last_model))

        last_model = best_model.replace(str(best_epoch).zfill(4), str(i).zfill(4))
        torch.save(model.state_dict(), os.path.join(folder, last_model))

        if valid_loss < best_valid_loss:
            try:
                os.remove(os.path.join(folder, best_model))
            except OSError:
                pass
            best_model = best_model.replace(str(best_epoch).zfill(4), str(i).zfill(4))
            torch.save(model.state_dict(), os.path.join(folder, best_model))
            best_epoch = i
            printdt("model saved")
            best_valid_loss = valid_loss
