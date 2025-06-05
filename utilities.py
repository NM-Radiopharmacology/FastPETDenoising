import datetime
import os
from torch import nn
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import sys
from tqdm import tqdm
import json
import itk
import SimpleITK as sitk
from data_augmentation_utilities import augment


def print2(*args):
    print(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), *args)


def patch_to_tensor(patch):

    patch = np.expand_dims(patch, axis=-1)
    patch = np.transpose(patch, (3, 0, 1, 2)).astype(float)

    return torch.Tensor(patch)


def denoise_patch(vol, model, device='cuda'):

    if torch.is_tensor(vol):
        arr = vol.detach().cpu().numpy()
        arr = np.asarray(arr.squeeze(0))
    elif isinstance(vol, np.ndarray):
        arr = vol
    else:
        raise ValueError('Input volume must either be a torch.Tensor or a numpy.ndarray.')

    patch = patch_to_tensor(vol)
    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
    den = den.detach().cpu().squeeze(0)
    den_arr = np.asarray(den.squeeze(0))

    return den_arr


def denoise_volume_gaussian(vol, model, overlap=None, patch_size=128, device='cuda'):

    if torch.is_tensor(vol):
        arr = vol.detach().cpu().numpy()
        arr = np.asarray(arr.squeeze(0))
    elif isinstance(vol, np.ndarray):
        arr = vol
    else:
        raise ValueError('Input volume must either be a torch.Tensor or a numpy.ndarray.')

    # Gaussian kernel
    ax = np.arange(-(patch_size // 2), patch_size // 2)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    xx = xx + 0.5
    yy = yy + 0.5
    zz = zz + 0.5
    sigma = patch_size / 4
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    # AS THERE WILL BE OVERLAP BETWEEN PATCHES, TWO EMPTY ARRAYS ARE CREATED BEFOREHAND: ONE TO KEEP SUMMING THE VALUES
    # OUTPUT BY THE NETWORK, IN THEIR RESPECTIVE INDICES, AND ONE TO COUNT HOW MANY TIMES EACH VOXEL IS FED INTO THE
    # NETWORK. IN THE END, THE DIVISION BETWEEN THE FORMER AND THE LATTER WILL PROVIDE THE WHOLE VOLUME PROCESSED BY THE
    # MODEL IN PATCHES.
    den_arr = np.zeros(arr.shape)       # EMPTY ARRAY IN WHICH THE NETWORK'S OUTPUT WILL BE STORED (BY SUM)
    counter_arr = np.zeros(arr.shape)   # EMPTY ARRAY TO COUNT THE NUMBER OF TIMES EACH VOXEL IS FED INTO THE NETWORK

    depth = arr.shape[0]  # z; axial
    height = arr.shape[1]  # y; coronal
    width = arr.shape[2]  # x; sagittal

    if overlap is None:
        # NUMBER OF PATCHES NECESSARY TO COVER THE WHOLE-VOLUME IN EACH OF THE THREE MAIN DIRECTIONS
        rz = int(np.ceil(depth / patch_size))
        ry = int(np.ceil(height / patch_size))
        rx = int(np.ceil(width / patch_size))

        # TOTAL NUMBER OF VOXELS THAT WILL BE OVERLAPPING IN EACH OF THE THREE MAIN DIRECTIONS
        sz = patch_size * rz - depth
        sy = patch_size * ry - height
        sx = patch_size * rx - width

        # TOTAL NUMBER OF VOXELS OVERLAPPING BETWEEN EACH PAIR OF CONSECUTIVE PATCHES (EXCEPT THE LAST ONE)
        try:
            divz = int(np.ceil(sz / (rz - 1)))
        except ZeroDivisionError:
            divz = 0
        if divz == 1 and rz > 2:
            rz = rz + 1
            sz = patch_size * rz - depth
            divz = int(np.ceil(sz / (rz - 1)))
        try:
            divy = int(np.ceil(sy / (ry - 1)))
        except ZeroDivisionError:
            divy = 0
        if divy == 1 and ry > 2:
            ry = ry + 1
            sy = patch_size * ry - height
            divy = int(np.ceil(sy / (ry - 1)))
        try:
            divx = int(np.ceil(sx / (rx - 1)))
        except ZeroDivisionError:
            divx = 0
        if divx == 1 and rx > 2:
            rx = rx + 1
            sx = patch_size * rx - width
            divx = int(np.ceil(sx / (rx - 1)))

    elif 0 < overlap < 1:

        div = int(np.ceil(patch_size * overlap))

        rz = int(np.ceil((depth - div) / (patch_size - div)))
        ry = int(np.ceil((height - div) / (patch_size - div)))
        rx = int(np.ceil((width - div) / (patch_size - div)))

        divx = div
        divy = div
        divz = div

    else:
        raise ValueError('Overlap fraction must be a float in the interval ]0, 1[.')

    # TRAVERSING THE IMAGE IN EACH DIRECTION, FEEDING THE PATCHES INTO THE MODEL
    start_z = 0
    for z in range(1, rz + 1):

        start_y = 0
        for y in range(1, ry + 1):

            start_x = 0
            for x in range(1, rx + 1):

                if x != rx and y != ry and z != rz:
                    patch = arr[start_z:start_z + patch_size, start_y:start_y + patch_size,
                            start_x:start_x + patch_size]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[start_z:start_z + patch_size, start_y:start_y + patch_size,
                    start_x:start_x + patch_size] += den * kernel
                    counter_arr[start_z:start_z + patch_size, start_y:start_y + patch_size,
                    start_x:start_x + patch_size] += kernel

                elif x == rx and y != ry and z != rz:
                    patch = arr[start_z:start_z + patch_size, start_y:start_y + patch_size, -patch_size:]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[start_z:start_z + patch_size, start_y:start_y + patch_size,
                    -patch_size:] += den * kernel
                    counter_arr[start_z:start_z + patch_size, start_y:start_y + patch_size, -patch_size:] += kernel

                elif x != rx and y == ry and z != rz:
                    patch = arr[start_z:start_z + patch_size, -patch_size:, start_x:start_x + patch_size]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[start_z:start_z + patch_size, -patch_size:,
                    start_x:start_x + patch_size] += den * kernel
                    counter_arr[start_z:start_z + patch_size, -patch_size:, start_x:start_x + patch_size] += kernel

                elif z == rz and y != ry and x != rx:
                    patch = arr[-patch_size:, start_y:start_y + patch_size, start_x:start_x + patch_size]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[-patch_size:, start_y:start_y + patch_size,
                    start_x:start_x + patch_size] += den * kernel
                    counter_arr[-patch_size:, start_y:start_y + patch_size, start_x:start_x + patch_size] += kernel

                elif z != rz and y == ry and x == rx:
                    patch = arr[start_z:start_z + patch_size, -patch_size:, -patch_size:]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[start_z:start_z + patch_size, -patch_size:, -patch_size:] += den * kernel
                    counter_arr[start_z:start_z + patch_size, -patch_size:, -patch_size:] += kernel

                elif z == rz and y != ry and x == rx:
                    patch = arr[-patch_size:, start_y:start_y + patch_size, -patch_size:]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[-patch_size:, start_y:start_y + patch_size, -patch_size:] += den * kernel
                    counter_arr[-patch_size:, start_y:start_y + patch_size, -patch_size:] += kernel

                elif z == rz and y == ry and x != rx:
                    patch = arr[-patch_size:, -patch_size:, start_x:start_x + patch_size]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[-patch_size:, -patch_size:, start_x:start_x + patch_size] += den * kernel
                    counter_arr[-patch_size:, -patch_size:, start_x:start_x + patch_size] += kernel

                elif z == rz and y == ry and x == rx:
                    patch = arr[-patch_size:, -patch_size:, -patch_size:]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[-patch_size:, -patch_size:, -patch_size:] += den * kernel
                    counter_arr[-patch_size:, -patch_size:, -patch_size:] += kernel

                start_x = (patch_size - divx) * x
            start_y = (patch_size - divy) * y
        start_z = (patch_size - divz) * z

    den_arr = den_arr / counter_arr

    return den_arr


class CrossCorrLoss(nn.Module):
    """
    Implementation of pearson correlation coefficient (cross correlation) as a loss function.
    The best correlation means maximizing the coefficient --> minimizing the loss means maximizing
    the coefficient ot minimizing the negative coefficient.
    """

    def __init__(self):
        super(CrossCorrLoss, self).__init__()

    def forward(self, output, target):
        x = output
        y = target

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        return -(torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))


class L1NormalizedLoss(nn.Module):
    """
    Implementation of the L1 loss function, with a normalization factor (real value + prediction + 1).
    """

    def __init__(self):
        super(L1NormalizedLoss, self).__init__()

    def forward(self, output, target):
        x = output
        y = target

        return torch.mean(torch.abs(x-y)/(x+y+1))


def kfold_cross_validation(k, data):

    validation_folds = {}

    if len(data) >= k:
        random.shuffle(data)
        fold_size = int(np.ceil(len(data)/3))
        start = 0
        for i in range(k-1):
            validation_folds[f"fold{i+1}"] = data[start:start+fold_size]
            start = start + fold_size
        validation_folds[f"fold{k}"] = data[-(len(data)-start):]

    else:
        print2("please provide enough training pairs for k-fold cross-validation")

    return validation_folds


def check_dataset_integrity(training_pairs):

    spacing = None
    validation_image_size = None

    different_spacings = []
    mismatch_image_reference = False

    print2('checking dataset integrity...')
    for img, ref in tqdm(training_pairs, file=sys.stdout, desc=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")):

        img_itk = itk.imread(img)
        ref_itk = itk.imread(ref)

        img_shape = np.asarray(img_itk).shape
        img_spacing = dict(img_itk)['spacing']

        if (np.array_equal(img_spacing, dict(ref_itk)['spacing']) and
                np.array_equal(img_shape, np.asarray(ref_itk).shape)):

            if spacing is None:
                spacing = img_spacing
                different_spacings.append(list(img_spacing))
            else:
                if not np.array_equal(img_spacing, spacing):
                    if list(img_spacing) not in different_spacings:
                        different_spacings.append(list(img_spacing))

            if validation_image_size is None:
                validation_image_size = np.asarray(img_shape)
            else:
                for i in range(len(img_shape)):
                    if img_shape[i] < validation_image_size[i]:
                        validation_image_size[i] = img_shape[i]

        else:
            mismatch_image_reference = True

    if len(different_spacings) > 1:
        print2("Warning! Spacing must match among the dataset! "
               f"The following spacings were found: {different_spacings}. "
               "Please resample all images to the same spacing.")

    if mismatch_image_reference:
        print2("Warning! Training images and reference images must match in size and spacing! "
               "Mismatches were found and excluded from the dataset.")

    return spacing, validation_image_size


def get_dataset(training_pairs=None, path_to_training=None, path_to_reference=None):

    if training_pairs is None:
        training_pairs = []
        if path_to_training is None or path_to_reference is None:
            path_to_reference = input("Enter path to the reference images (ordered): ")
            path_to_training = input("Enter path to the training images (ordered): ")

            if len(os.listdir(path_to_training)) == len(os.listdir(path_to_reference)):
                for img, ref in zip(sorted(os.listdir(path_to_training)), sorted(os.listdir(path_to_reference))):
                    img = os.path.join(path_to_training, img)
                    ref = os.path.join(path_to_reference, ref)

                    training_pairs.append([img, ref])

            else:
                print2("reference images must match training images!")
                return

    spacing, validation_image_size = check_dataset_integrity(training_pairs)

    dataset = {
        "training_pairs": training_pairs,
        "validation_image_size": validation_image_size.astype(int).tolist(),
        "spacing": spacing.astype(float).tolist()
    }

    return dataset


def create_configuration_file(default_params=True, training_pairs=None):
    config_file = "configuration.json"

    configuration = {
        "datetime": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "dataset": get_dataset(training_pairs),
        "patch_size": None,
        "network_configuration": None,
        "loss_function": None,
        "batch_size": None,
        "optimizer": None,
        "N_epochs": None,
        "validation_every_N_epochs": None,
        "learning_rate": None,
        "learning_rate_decay_rate": None,
        "learning_rate_decay_steps": None,
        "perform_data_augmentation": None,
        "training_set_fraction": 0.25
    }

    while configuration["network_configuration"] not in ["3D", "2.5D-1channel", "2.5D-3channel"]:
        configuration["network_configuration"] =\
            input("Enter network configuration (3D, 2.5D-1channel, 2.5D-3channel): ")

    if not default_params:

        is_patch_size_defined = False
        while not is_patch_size_defined:
            patch_size_input = input("Enter patch size in format (z,y,x) (e.g. (128,128,128)): ")
            patch_size_input.replace(' ', '')
            patch_size_input = patch_size_input[1:-1].split(',')
            try:
                patch_size = [int(s) for s in patch_size_input]
                is_patch_size_defined = True
                configuration["patch_size"] = patch_size
            except ValueError:
                pass

        while configuration["loss_function"] not in ["MSE", "L1", "normalised_L1", "CrossCorr"]:
            configuration["loss_function"] = input("Enter loss function (MSE, L1, normalised_L1, CrossCorr): ")

        configuration["batch_size"] = int(input("Enter batch size for training: "))

        while configuration["optimizer"] not in ["Adam", "SGD"]:
            configuration["optimizer"] = input("Enter optimizer (Adam or SGD): ")

        configuration["N_epochs"] = int(input("Enter number of epochs for training: "))

        configuration["validation_every_N_epochs"] = int(input("Run validation every __ epochs: "))

        configuration["learning_rate"] = float(input("Enter learning rate for training: "))

        configuration["learning_rate_decay_rate"] = float(input("Enter learning rate decay rate (e.g. 0.9): "))

        configuration["learning_rate_decay_steps"] = int(input("Enter learning rate decay steps (e.g. 20): "))

        while configuration["perform_data_augmentation"] not in ['True', 'False']:
            configuration["perform_data_augmentation"] = (
                input("Enter whether to perform data augmentation (True/False): "))

        if configuration["perform_data_augmentation"].lower() == 'true':
            configuration["perform_data_augmentation"] = True
        elif configuration["perform_data_augmentation"].lower() == 'false':
            configuration["perform_data_augmentation"] = False

    else:

        configuration["patch_size"] = 128
        configuration["loss_function"] = "MSE"
        configuration["batch_size"] = 2 if configuration["network_configuration"] == "3D" else 144
        configuration["optimizer"] = "Adam"
        configuration["N_epochs"] = 1000
        configuration["validation_every_N_epochs"] = 20
        configuration["learning_rate"] = 0.01
        configuration["learning_rate_decay_rate"] = 0.9
        configuration["learning_rate_decay_steps"] = 50
        configuration["perform_data_augmentation"] = True

    configuration_json = json.dumps(configuration, indent=4)
    open(config_file, "w").close()
    with open(config_file, "w") as config_file:
        config_file.write(configuration_json)

    print2("configuration file created")

    return configuration


class MakeTorchDataset(Dataset):
    """
    Class that receives a pd.dataframe in which the first column contains the paths to the input images and the
    second contains the paths to the target images, and returns the images as torch tensors. Data augmentation
    is performed.
    """

    def __init__(self, df, augmentations, dim=None, validation=None, patch_size=None, validation_image_size=None):

        if dim is None:
            self.dim = 3

        if patch_size is None:
            patch_size = [128, 128, 128]

        if validation is None:
            validation = False

        if validation and validation_image_size is None:
            validation_image_size = [patch_size[0], patch_size[1], patch_size[2]] if dim == 3 else [patch_size[0],
                                                                                                    patch_size[1]]

        self.patch_size = patch_size
        self.validation = validation
        self.df = df
        self.validation_image_size = validation_image_size

        self.augmentations = augmentations  # method that performs the desired augmentations/transformations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        augmentations = self.augmentations
        patch_size = self.patch_size
        validation = self.validation
        validation_image_size = self.validation_image_size
        # Training pair: (input, target)
        # Column 0: paths to the input images
        image_path = self.df.iloc[idx, 0]
        # Column 1: paths do the target images
        target_path = self.df.iloc[idx, 1]

        # Check if it's not the validation set
        if not validation:
            # Check if augmentations must be performed or not
            if not augmentations:
                # Input image
                image = np.asarray(itk.imread(image_path))  # ATTENTION: path must not contain accents

                # -------------------------- testing new random patch generation (increased probability from the center)
                random_indices = []
                for dim in range(image.ndim):
                    cut = image.shape[dim] - patch_size[dim]

                    if cut > 0:
                        prob_arr_size = cut // 2 if cut % 2 != 0 else (cut - 1) // 2

                        # gaussian probability function draw
                        ax = np.arange(-(prob_arr_size // 2), prob_arr_size // 2 + 1)
                        sigma = prob_arr_size / 3
                        kernel = np.exp(-(ax ** 2) / (2 * sigma ** 2))
                        kernel = kernel / np.sum(kernel)

                        random_indices.append(np.random.choice(np.arange(len(kernel)), p=kernel))

                    elif cut == 0:  # if cut = 0, patch size is equal to image size
                        random_indices.append(0)  # NOT TESTED YET

                    else:  # if cut < 0, patch size is larger than image size
                        print("patch size larger than image size!")
                        return 1  # NOT TESTED YET

                z, y, x = random_indices[0], random_indices[1], random_indices[2]
                # ------------------------------------------------------------------------------------------------------

                # Transforming size (D, W, H) to (D, W, H, C) where C is the number of channels
                image = image[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]]

                # Target image
                target = np.asarray(itk.imread(target_path))  # ATTENTION: path must not contain accents
                # Transforming size (D, W, H) to (D, W, H, C) where C is the number of channels
                target = target[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]]

            else:   # augmentations = True
                # Input image
                image = sitk.ReadImage(image_path)  # ATTENTION: path must not contain accents
                shape = sitk.GetArrayFromImage(image).shape
                # Target image
                target = sitk.ReadImage(target_path)  # ATTENTION: path must not contain accents

                # TRANSFORMS
                image, target = augment(image, target, shape)

        else:  # if validation = True

            image = np.asarray(itk.imread(image_path))  # ATTENTION: path must not contain accents
            target = np.asarray(itk.imread(target_path))  # ATTENTION: path must not contain accents

            lower_cut = []
            for dim in range(len(validation_image_size)):
                lower_cut.append(
                    0 if validation_image_size[dim] == image.shape[dim] else
                    int(np.floor((image.shape[dim] - validation_image_size[dim]) / 2))
                )

            image = image[lower_cut[0]:lower_cut[0] + validation_image_size[0],
                    lower_cut[1]:lower_cut[1] + validation_image_size[1],
                    lower_cut[2]:lower_cut[2] + validation_image_size[2]]
            target = target[lower_cut[0]:lower_cut[0] + validation_image_size[0],
                     lower_cut[1]:lower_cut[1] + validation_image_size[1],
                     lower_cut[2]:lower_cut[2] + validation_image_size[2]]

        image = np.expand_dims(image, axis=-1)
        target = np.expand_dims(target, axis=-1)

        # (D, W, H, C) to (C, D, W, H) <=> (0, 1, 2, 3) to (3, 0, 1, 2)
        image = np.transpose(image, (3, 0, 1, 2)).astype(float)
        target = np.transpose(target, (3, 0, 1, 2)).astype(float)

        # Numpy to torch tensor
        image = torch.Tensor(image)
        target = torch.Tensor(target)

        return image, target

