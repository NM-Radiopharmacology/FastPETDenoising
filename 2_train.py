"""
script to train the network
training is performed from the configuration.json file in the project's directory
run 1_define_training_configuration.py to configure training and create configuration.json accordingly
"""
import os
import numpy as np
import pandas as pd
from utilities import printdt, CrossCorrLoss, L1NormalizedLoss, denoise_volume_gaussian, denoise_patch, MakeTorchDataset
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
                arr = np.asarray(arr.squeeze(0))

                if list(arr.shape) != list(patch_size):

                    den = denoise_volume_gaussian(arr, model, patch_size=patch_size)
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

if len(patch_size) == 1:
    patch_size = [patch_size, patch_size, patch_size]

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

    training_lst = [pair for pair in training_pairs if pair not in validation_folds[fold]]
    validation_lst = validation_folds[fold]

    printdt(f'training set size: {len(training_lst)}')
    printdt(f'validation set size: {len(validation_lst)}')

    train_df = pd.DataFrame(training_lst, columns=['images', 'targets'])
    training_set = MakeTorchDataset(train_df, patch_size=patch_size, augmentations=False)

    valid_df = pd.DataFrame(validation_lst, columns=['images', 'targets'])

    printdt("batching validation set")
    validation_set = MakeTorchDataset(valid_df, augmentations=False, patch_size=patch_size,
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
    elif network_configuration == "2.5D-1channel":
        model = UNet25D(in_channels=1)
        input_size = (1, patch_size[0], patch_size[1])
    elif network_configuration == "2.5D-3channel":
        model = UNet25D(in_channels=3)
        input_size = (1, patch_size[0], patch_size[1])
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
