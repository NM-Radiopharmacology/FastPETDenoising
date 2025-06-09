import datetime
import os
import torch
import numpy as np
import random
import sys
from tqdm import tqdm
import json
import itk


def printdt(*args):
    print(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), *args)


def patch_to_tensor(patch):
    if len(patch.shape) == 3:
        patch = np.expand_dims(patch, axis=-1)
        patch = np.transpose(patch, (3, 0, 1, 2)).astype(float)
    else:
        patch = np.expand_dims(patch, axis=-1)
        patch = np.transpose(patch, (2, 0, 1)).astype(float)
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


def denoise_slice_gaussian(slc, model, overlap=None, patch_size=None, device='cuda'):
    if patch_size is None:
        patch_size = [144, 144]
    if torch.is_tensor(slc):
        arr = slc.detach().cpu().numpy()
        arr = np.asarray(arr.squeeze(0))
    elif isinstance(slc, np.ndarray):
        arr = slc
    else:
        raise ValueError('Input volume must either be a torch.Tensor or a numpy.ndarray.')

    # 2D Gaussian kernel
    ax_y = np.arange(-(patch_size[0] // 2), patch_size[0] // 2) + 0.5
    ax_x = np.arange(-(patch_size[1] // 2), patch_size[1] // 2) + 0.5
    yy, xx = np.meshgrid(ax_y, ax_x, indexing='ij')
    sigma_y = patch_size[0] / 4
    sigma_x = patch_size[1] / 4
    kernel = np.exp(-((xx ** 2) / (2 * sigma_x ** 2) + (yy ** 2) / (2 * sigma_y ** 2)))
    kernel /= np.sum(kernel)

    dim = arr.ndim

    if dim == 3:   # 3-channel axial slice
        den_arr = np.zeros([arr.shape[2], arr.shape[1]])
        counter_arr = np.zeros([arr.shape[2], arr.shape[1]])
        height = arr.shape[1]  # y
        width = arr.shape[2]  # x
    else:
        # Init output and counter arrays
        den_arr = np.zeros(arr.shape)
        counter_arr = np.zeros(arr.shape)
        height = arr.shape[0]  # y
        width = arr.shape[1]  # x

    if overlap is None:
        ry = int(np.ceil(height / patch_size[0]))
        rx = int(np.ceil(width / patch_size[1]))

        sy = patch_size[0] * ry - height
        sx = patch_size[1] * rx - width

        try:
            divy = int(np.ceil(sy / (ry - 1)))
        except ZeroDivisionError:
            divy = 0
        if divy == 1 and ry > 2:
            ry += 1
            sy = patch_size[0] * ry - height
            divy = int(np.ceil(sy / (ry - 1)))

        try:
            divx = int(np.ceil(sx / (rx - 1)))
        except ZeroDivisionError:
            divx = 0
        if divx == 1 and rx > 2:
            rx += 1
            sx = patch_size[1] * rx - width
            divx = int(np.ceil(sx / (rx - 1)))

    elif 0 < overlap < 1:
        divy = int(np.ceil(patch_size[0] * overlap))
        divx = int(np.ceil(patch_size[1] * overlap))

        ry = int(np.ceil((height - divy) / (patch_size[0] - divy)))
        rx = int(np.ceil((width - divx) / (patch_size[1] - divx)))
    else:
        raise ValueError('Overlap fraction must be a float in the interval ]0, 1[.')

    # Iterate over patches
    start_y = 0
    for y in range(1, ry + 1):
        start_x = 0
        for x in range(1, rx + 1):
            patch = None
            den_slice_coordinates = None
            # Calculate patch location
            if y != ry and x != rx:
                if dim == 3:
                    patch = arr[:, start_y:start_y + patch_size[0], start_x:start_x + patch_size[1]]
                else:
                    patch = arr[start_y:start_y + patch_size[0], start_x:start_x + patch_size[1]]
                den_slice_coordinates = (slice(start_y, start_y + patch_size[0]),
                                         slice(start_x, start_x + patch_size[1]))
            elif y == ry and x != rx:
                if dim == 3:
                    patch = arr[:, -patch_size[0]:, start_x:start_x + patch_size[1]]
                else:
                    patch = arr[-patch_size[0]:, start_x:start_x + patch_size[1]]
                den_slice_coordinates = (slice(-patch_size[0], None), slice(start_x, start_x + patch_size[1]))
            elif y != ry and x == rx:
                if dim == 3:
                    patch = arr[:, start_y:start_y + patch_size[0], -patch_size[1]:]
                else:
                    patch = arr[start_y:start_y + patch_size[0], -patch_size[1]:]
                den_slice_coordinates = (slice(start_y, start_y + patch_size[0]), slice(-patch_size[1], None))
            elif y == ry and x == rx:
                if dim == 3:
                    patch = arr[:, -patch_size[0]:, -patch_size[1]:]
                else:
                    patch = arr[-patch_size[0]:, -patch_size[1]:]
                den_slice_coordinates = (slice(-patch_size[0], None), slice(-patch_size[1], None))

            # Inference and accumulate
            if dim == 3:
                patch_tensor = torch.Tensor(patch)
            else:
                patch_tensor = patch_to_tensor(patch)
            den = model(patch_tensor.to(device).unsqueeze(0))  # (c, h, w) -> (1, c, h, w)
            den = den.detach().cpu().squeeze(0)
            den = np.asarray(den.squeeze(0))
            den_arr[den_slice_coordinates] += den * kernel
            counter_arr[den_slice_coordinates] += kernel

            start_x = (patch_size[1] - divx) * x
        start_y = (patch_size[0] - divy) * y

    # Final normalized output
    den_arr = den_arr / counter_arr

    return den_arr


def denoise_volume_gaussian(vol, model, overlap=None, patch_size=None, device='cuda'):
    if patch_size is None:
        patch_size = [128, 128, 128]
    if torch.is_tensor(vol):
        arr = vol.detach().cpu().numpy()
        arr = np.asarray(arr.squeeze(0))
    elif isinstance(vol, np.ndarray):
        arr = vol
    else:
        raise ValueError('Input volume must either be a torch.Tensor or a numpy.ndarray.')

    # Gaussian kernel
    ax_z = np.arange(-(patch_size[0] // 2), patch_size[0] // 2) + 0.5
    ax_y = np.arange(-(patch_size[1] // 2), patch_size[1] // 2) + 0.5
    ax_x = np.arange(-(patch_size[2] // 2), patch_size[2] // 2) + 0.5
    zz, yy, xx = np.meshgrid(ax_z, ax_y, ax_x, indexing='ij')
    sigma_z = patch_size[0] / 4
    sigma_y = patch_size[1] / 4
    sigma_x = patch_size[2] / 4
    kernel = np.exp(-(
            (xx ** 2) / (2 * sigma_x ** 2) +
            (yy ** 2) / (2 * sigma_y ** 2) +
            (zz ** 2) / (2 * sigma_z ** 2)
    ))  # Compute anisotropic 3D Gaussian kernel
    kernel /= np.sum(kernel)  # Normalize to sum to 1

    # AS THERE WILL BE OVERLAP BETWEEN PATCHES, TWO EMPTY ARRAYS ARE CREATED BEFOREHAND: ONE TO KEEP SUMMING THE VALUES
    # OUTPUT BY THE NETWORK, IN THEIR RESPECTIVE INDICES, AND ONE TO COUNT HOW MANY TIMES EACH VOXEL IS FED INTO THE
    # NETWORK. IN THE END, THE DIVISION BETWEEN THE FORMER AND THE LATTER WILL PROVIDE THE WHOLE VOLUME PROCESSED BY THE
    # MODEL IN PATCHES.
    den_arr = np.zeros(arr.shape)  # EMPTY ARRAY IN WHICH THE NETWORK'S OUTPUT WILL BE STORED (BY SUM)
    counter_arr = np.zeros(arr.shape)  # EMPTY ARRAY TO COUNT THE NUMBER OF TIMES EACH VOXEL IS FED INTO THE NETWORK

    depth = arr.shape[0]  # z; axial
    height = arr.shape[1]  # y; coronal
    width = arr.shape[2]  # x; sagittal

    if overlap is None:
        # NUMBER OF PATCHES NECESSARY TO COVER THE WHOLE-VOLUME IN EACH OF THE THREE MAIN DIRECTIONS
        rz = int(np.ceil(depth / patch_size[0]))
        ry = int(np.ceil(height / patch_size[1]))
        rx = int(np.ceil(width / patch_size[2]))

        # TOTAL NUMBER OF VOXELS THAT WILL BE OVERLAPPING IN EACH OF THE THREE MAIN DIRECTIONS
        sz = patch_size[0] * rz - depth
        sy = patch_size[1] * ry - height
        sx = patch_size[2] * rx - width

        # TOTAL NUMBER OF VOXELS OVERLAPPING BETWEEN EACH PAIR OF CONSECUTIVE PATCHES (EXCEPT THE LAST ONE)
        try:
            divz = int(np.ceil(sz / (rz - 1)))
        except ZeroDivisionError:
            divz = 0
        if divz == 1 and rz > 2:
            rz = rz + 1
            sz = patch_size[0] * rz - depth
            divz = int(np.ceil(sz / (rz - 1)))
        try:
            divy = int(np.ceil(sy / (ry - 1)))
        except ZeroDivisionError:
            divy = 0
        if divy == 1 and ry > 2:
            ry = ry + 1
            sy = patch_size[1] * ry - height
            divy = int(np.ceil(sy / (ry - 1)))
        try:
            divx = int(np.ceil(sx / (rx - 1)))
        except ZeroDivisionError:
            divx = 0
        if divx == 1 and rx > 2:
            rx = rx + 1
            sx = patch_size[2] * rx - width
            divx = int(np.ceil(sx / (rx - 1)))

    elif 0 < overlap < 1:  # not tested!!

        divz = int(np.ceil(patch_size[0] * overlap))
        divy = int(np.ceil(patch_size[1] * overlap))
        divx = int(np.ceil(patch_size[2] * overlap))

        rz = int(np.ceil((depth - divz) / (patch_size[0] - divz)))
        ry = int(np.ceil((height - divy) / (patch_size[1] - divy)))
        rx = int(np.ceil((width - divx) / (patch_size[2] - divx)))

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
                    patch = arr[start_z:start_z + patch_size[0], start_y:start_y + patch_size[1],
                            start_x:start_x + patch_size[2]]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[start_z:start_z + patch_size[0], start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2]] += den * kernel
                    counter_arr[start_z:start_z + patch_size[0], start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2]] += kernel

                elif x == rx and y != ry and z != rz:
                    patch = arr[start_z:start_z + patch_size[0], start_y:start_y + patch_size[1], -patch_size[2]:]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[start_z:start_z + patch_size[0], start_y:start_y + patch_size[1],
                    -patch_size[2]:] += den * kernel
                    counter_arr[start_z:start_z + patch_size[0], start_y:start_y + patch_size[1],
                    -patch_size[2]:] += kernel

                elif x != rx and y == ry and z != rz:
                    patch = arr[start_z:start_z + patch_size[0], -patch_size[1]:, start_x:start_x + patch_size[2]]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[start_z:start_z + patch_size[0], -patch_size[1]:,
                    start_x:start_x + patch_size[2]] += den * kernel
                    counter_arr[start_z:start_z + patch_size[0], -patch_size[1]:,
                    start_x:start_x + patch_size[2]] += kernel

                elif z == rz and y != ry and x != rx:
                    patch = arr[-patch_size[0]:, start_y:start_y + patch_size[1], start_x:start_x + patch_size[2]]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[-patch_size[0]:, start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2]] += den * kernel
                    counter_arr[-patch_size[0]:, start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2]] += kernel

                elif z != rz and y == ry and x == rx:
                    patch = arr[start_z:start_z + patch_size[0], -patch_size[1]:, -patch_size[2]:]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[start_z:start_z + patch_size[0], -patch_size[1]:, -patch_size[2]:] += den * kernel
                    counter_arr[start_z:start_z + patch_size[0], -patch_size[1]:, -patch_size[2]:] += kernel

                elif z == rz and y != ry and x == rx:
                    patch = arr[-patch_size[0]:, start_y:start_y + patch_size[1], -patch_size[2]:]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[-patch_size[0]:, start_y:start_y + patch_size[1], -patch_size[2]:] += den * kernel
                    counter_arr[-patch_size[0]:, start_y:start_y + patch_size[1], -patch_size[2]:] += kernel

                elif z == rz and y == ry and x != rx:
                    patch = arr[-patch_size[0]:, -patch_size[1]:, start_x:start_x + patch_size[2]]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[-patch_size[0]:, -patch_size[1]:, start_x:start_x + patch_size[2]] += den * kernel
                    counter_arr[-patch_size[0]:, -patch_size[1]:, start_x:start_x + patch_size[2]] += kernel

                elif z == rz and y == ry and x == rx:
                    patch = arr[-patch_size[0]:, -patch_size[1]:, -patch_size[2]:]
                    patch = patch_to_tensor(patch)
                    den = model(patch.to(device).unsqueeze(0))  # (c, h, w) ----.unsqueeze(0)---> (b, c, h, w)
                    den = den.detach().cpu().squeeze(0)
                    den = np.asarray(den.squeeze(0))
                    den_arr[-patch_size[0]:, -patch_size[1]:, -patch_size[2]:] += den * kernel
                    counter_arr[-patch_size[0]:, -patch_size[1]:, -patch_size[2]:] += kernel

                start_x = (patch_size[2] - divx) * x
            start_y = (patch_size[1] - divy) * y
        start_z = (patch_size[0] - divz) * z

    den_arr = den_arr / counter_arr

    return den_arr


def kfold_cross_validation(k, data):
    validation_folds = {}

    if len(data) >= k:
        random.shuffle(data)
        fold_size = int(np.ceil(len(data) / 3))
        start = 0
        for i in range(k - 1):
            validation_folds[f"fold{i + 1}"] = data[start:start + fold_size]
            start = start + fold_size
        validation_folds[f"fold{k}"] = data[-(len(data) - start):]

    else:
        printdt("please provide enough training pairs for k-fold cross-validation")

    return validation_folds


def check_dataset_integrity(training_pairs):
    spacing = None
    validation_image_size = None

    different_spacings = []
    mismatch_image_reference = False

    printdt('checking dataset integrity...')
    for img, ref in tqdm(training_pairs, file=sys.stdout, desc=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")):

        img_itk = itk.imread(img)
        ref_itk = itk.imread(ref)

        img_shape = np.asarray(img_itk).shape
        img_spacing = dict(img_itk)['spacing']

        if len(img_shape) != 3:
            print(f"All images must be 3D! (Found {img_shape} shape)")
            sys.exit()

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
        printdt("Warning! Spacing must match among the dataset! "
                f"The following spacings were found: {different_spacings}. "
                "Please resample all images to the same spacing.")

    if mismatch_image_reference:
        printdt("Warning! Training images and reference images must match in size and spacing! "
                "Mismatches were found and excluded from the dataset.")

    return spacing, validation_image_size


def get_dataset(training_pairs=None, path_to_training=None, path_to_reference=None):
    if training_pairs is None:
        training_pairs = []
        if path_to_training is None or path_to_reference is None:
            path_to_training = input("Enter path to the training images: ")
            path_to_reference = input("Enter path to the respective reference images (must be ordered accordingly!): ")

            if len(os.listdir(path_to_training)) == len(os.listdir(path_to_reference)):
                for img, ref in zip(sorted(os.listdir(path_to_training)), sorted(os.listdir(path_to_reference))):
                    img = os.path.join(path_to_training, img)
                    ref = os.path.join(path_to_reference, ref)

                    training_pairs.append([img, ref])

            else:
                printdt("reference images must match training images!")
                return

    spacing, validation_image_size = check_dataset_integrity(training_pairs)

    dataset = {
        "training_pairs": training_pairs,
        "validation_image_size": validation_image_size.astype(int).tolist(),
        "spacing": spacing.astype(float).tolist()
    }

    return dataset


def create_configuration_file():
    config_file = "manuscript_model/configuration.json"

    configuration = {
        "datetime": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "dataset": get_dataset(),
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
        "training_set_fraction": None
    }

    while configuration["network_configuration"] not in ["3D", "1channel-2.5D", "3channel-2.5D"]:
        configuration["network_configuration"] = \
            input("Enter network configuration (3D, 1channel-2.5D, 3channel-2.5D): ")

    # Would you like to use standard training parameters? Set to False to introduce manually
    use_default_training_params = None
    while use_default_training_params not in ['true', 'false']:
        use_default_training_params = (
            input("Enter whether to use default training parameters (True) or to set them manually (False): ")).lower()
    if use_default_training_params == 'true':
        use_default_training_params = True
    else:
        use_default_training_params = False
    if not use_default_training_params:

        is_patch_size_defined = False
        while not is_patch_size_defined:
            patch_size_input = input("Enter patch size in format (z,y,x) (e.g. (128,128,128)): ")
            patch_size_input.replace(' ', '')
            patch_size_input = patch_size_input[1:-1].split(',')
            try:
                patch_size = [int(s) for s in patch_size_input]
                if ((configuration["network_configuration"] == '3D' and len(patch_size) == 3) or
                        ('2.5D' in configuration["network_configuration"] and len(patch_size) == 2)):
                    is_patch_size_defined = True
                    configuration["patch_size"] = patch_size
                else:
                    printdt(f"Invalid patch size ({patch_size}) for a {configuration['network_configuration']} network "
                            f"configuration!")
            except ValueError:
                pass

        while configuration["loss_function"] not in ["MSE", "L1", "normalised_L1", "CrossCorr"]:
            configuration["loss_function"] = input("Enter loss function (MSE, L1, normalised_L1, CrossCorr): ")

        configuration["batch_size"] = int(input("Enter batch size for training: "))

        while configuration["optimizer"] not in ["Adam", "SGD"]:
            configuration["optimizer"] = input("Enter optimizer (Adam or SGD): ")

        configuration["N_epochs"] = int(input("Enter number of epochs for training: "))

        configuration["validation_every_N_epochs"] = int(input("Run validation every __ epochs (e.g. 20): "))

        configuration["learning_rate"] = float(input("Enter learning rate for training: "))

        configuration["learning_rate_decay_rate"] = float(input("Enter learning rate decay rate (e.g. 0.9): "))

        configuration["learning_rate_decay_steps"] = int(input("Make learning rate decay every __ epochs (e.g. 50): "))

        while configuration["perform_data_augmentation"] not in ['True', 'False']:
            configuration["perform_data_augmentation"] = (
                input("Enter whether to perform data augmentation (True/False): "))

        if configuration["perform_data_augmentation"].lower() == 'true':
            configuration["perform_data_augmentation"] = True
        elif configuration["perform_data_augmentation"].lower() == 'false':
            configuration["perform_data_augmentation"] = False

    else:

        configuration["patch_size"] = [128, 128, 128] if configuration["network_configuration"] == "3D" else [144, 144]
        configuration["loss_function"] = "MSE"
        configuration["batch_size"] = 1 if configuration["network_configuration"] == "3D" else 64
        configuration["optimizer"] = "Adam"
        configuration["N_epochs"] = 1000
        configuration["validation_every_N_epochs"] = 20
        configuration["learning_rate"] = 0.01
        configuration["learning_rate_decay_rate"] = 0.9
        configuration["learning_rate_decay_steps"] = 50
        configuration["perform_data_augmentation"] = True

    configuration["training_set_fraction"] = 0.25 if configuration["network_configuration"] == "3D" else 0.75

    configuration_json = json.dumps(configuration, indent=4)
    open(config_file, "w").close()
    with open(config_file, "w") as config_file:
        config_file.write(configuration_json)

    printdt("configuration file created")

    return configuration

