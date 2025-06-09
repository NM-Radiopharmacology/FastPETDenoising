from torch.utils.data import Dataset
import torch
import numpy as np
import itk
import SimpleITK as sitk
from data_augmentation_utilities import augment


class MakeTorchDataset(Dataset):
    """
    Class that receives a pd.dataframe in which the first column contains the paths to the input images and the
    second contains the paths to the target images, and returns the images as torch tensors. Data augmentation
    is performed.
    """

    def __init__(self, df, patch_size, augmentations, network_configuration,
                 validation=None, validation_image_size=None):

        if validation is None:
            validation = False

        if validation and validation_image_size is None:
            validation_image_size = [patch_size[0], patch_size[1], patch_size[2]] if len(patch_size) == 3 else\
                [patch_size[0], patch_size[1]]

        self.df = df
        self.patch_size = patch_size
        self.augmentations = augmentations
        self.network_configuration = network_configuration

        self.validation = validation
        self.validation_image_size = validation_image_size

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
        if len(list(self.df.columns)) > 2:
            anatomical_plane = self.df.iloc[idx, 2]
            i = self.df.iloc[idx, 3]
        else:
            anatomical_plane = None
            i = None

        # Check if it's not the validation set
        if not validation:
            # Check if augmentations must be performed or not
            if not augmentations:
                # Input image
                image = np.asarray(itk.imread(image_path))  # ATTENTION: path must not contain accents
                # Target image
                target = np.asarray(itk.imread(target_path))  # ATTENTION: path must not contain accents

                if i is not None:
                    if anatomical_plane == 'axial':         # z
                        if self.network_configuration == "3channel-2.5D":
                            image = image[i - 1:i + 2, :, :]
                            target = target[i - 1:i + 2, :, :]
                        else:
                            image = image[i, :, :]
                            target = target[i, :, :]
                    elif anatomical_plane == 'coronal':     # y
                        image = image[:, i, :]
                        target = target[:, i, :]
                    elif anatomical_plane == 'sagittal':    # x
                        image = image[:, :, i]
                        target = target[:, :, i]

            else:  # augmentations = True
                # Input image
                image = sitk.ReadImage(image_path)  # ATTENTION: path must not contain accents
                shape = sitk.GetArrayFromImage(image).shape
                # Target image
                target = sitk.ReadImage(target_path)  # ATTENTION: path must not contain accents

                # TRANSFORMS
                image, target = augment(image, target, shape)

                if i is not None:
                    if anatomical_plane == 'axial':         # z
                        if self.network_configuration == "3channel-2.5D":
                            image = image[i - 1:i + 2, :, :]
                            target = target[i - 1:i + 2, :, :]
                        else:
                            image = image[i, :, :]
                            target = target[i, :, :]
                    elif anatomical_plane == 'coronal':     # y
                        image = image[:, i, :]
                        target = target[:, i, :]
                    elif anatomical_plane == 'sagittal':    # x
                        image = image[:, :, i]
                        target = target[:, :, i]

            # ------------------------------ testing new random patch generation (increased probability from the center)
            random_indices = []
            for dim in range(image.ndim):
                if self.network_configuration == "3channel-2.5D":
                    if dim == 0:
                        continue
                    else:
                        cut = image.shape[dim] - patch_size[dim - 1]
                else:
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
                    #print("patch size larger than image size!")
                    image, target = None, None
                    return image, target  # NOT TESTED YET

            if image.ndim == 3:
                if self.network_configuration == "3channel-2.5D":
                    x, y = random_indices[0], random_indices[1]
                    image = image[:, y:y + patch_size[1], x:x + patch_size[0]]
                    target = target[1, y:y + patch_size[1], x:x + patch_size[0]]
                else:
                    z, y, x = random_indices[0], random_indices[1], random_indices[2]
                    image = image[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]]
                    target = target[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2]]

            else:
                x, y = random_indices[0], random_indices[1]
                image = image[x:x + patch_size[0], y:y + patch_size[1]]
                target = target[x:x + patch_size[0], y:y + patch_size[1]]
            # ----------------------------------------------------------------------------------------------------------

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

        if self.network_configuration != "3channel-2.5D":
            image = np.expand_dims(image, axis=-1)
            # (D, W, H, C) to (C, D, W, H) <=> (0, 1, 2, 3) to (3, 0, 1, 2)
            image = np.transpose(image, (3, 0, 1, 2)).astype(float) if image.ndim == 4 else (
                np.transpose(image, (2, 0, 1)).astype(float))

        target = np.expand_dims(target, axis=-1)
        target = np.transpose(target, (3, 0, 1, 2)).astype(float) if target.ndim == 4 else (
            np.transpose(target, (2, 0, 1)).astype(float))

        # Numpy to torch tensor
        image = torch.Tensor(image)
        target = torch.Tensor(target)

        return image, target
