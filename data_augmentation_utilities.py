import numpy as np
import SimpleITK as sitk
import random


def eul2quat(ax, ay, az, atol=1e-8):
    """
    Translate between Euler angle (ZYX) order and quaternion representation (versor) of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.
    """

    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r = np.zeros((3, 3))
    r[0, 0] = cz * cy
    r[0, 1] = cz * sy * sx - sz * cx
    r[0, 2] = cz * sy * cx + sz * sx

    r[1, 0] = sz * cy
    r[1, 1] = sz * sy * sx + cz * cx
    r[1, 2] = sz * sy * cx - cz * sx

    r[2, 0] = -sy
    r[2, 1] = cy * sx
    r[2, 2] = cy * cx

    # Compute quaternion:
    qs = 0.5 * np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs, 0.0, atol):
        i = np.argmax([r[0, 0], r[1, 1], r[2, 2]])
        j = (i + 1) % 3
        k = (j + 1) % 3
        w = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1)
        qv[i] = 0.5 * w
        qv[j] = (r[i, j] + r[j, i]) / (2 * w)
        qv[k] = (r[i, k] + r[k, i]) / (2 * w)
    else:
        denom = 4 * qs
        qv[0] = (r[2, 1] - r[1, 2]) / denom
        qv[1] = (r[0, 2] - r[2, 0]) / denom
        qv[2] = (r[1, 0] - r[0, 1]) / denom

    return qv


def augment_images_spatial_composite(original_image, reference_image, T0, T1, T2, T1_params, T2_params,
                                     interpolator=sitk.sitkLinear, default_intensity_value=0.0):
    """
    Generate the resampled images based on the given transformations - translation, rotation and skew.
    Args:
        original_image (SimpleITK image): The image to resample and transform.
        reference_image (SimpleITK image): The image onto which resample will be performed.
        T0 (SimpleITK transform): Transformation which maps points from the reference_image coordinate system
            to the original_image coordinate system.
        T1 (SimpleITK transform - Similarity3DTransform): similarity 3D transform with rotation as a versor, and
        isotropic scaling around a fixed center with translation. Used for translation and rotation.
        T1_params (List of lists): parameter values for the T1 transform.
        T2 (SimpleITK transform - ComposeScaleSkewVersor3DTransform): transform that applies a versor rotation and
         translation & scale/skew to the space. Used for anisotropic scaling.
        T2_params (List of lists): parameter values for the T2 transform.
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    """

    T1.SetParameters(T1_params)

    T2.SetRotation([0, 0, 1], 0)    # No rotation
    T2.SetScale(T2_params[0])
    T2.SetTranslation(T2_params[1])
    T2.SetSkew(T2_params[2])

    # Augmentation is done in the reference image space, so we first map the points from the reference image space
    # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
    T0 = sitk.Transform(T0)
    T1 = sitk.Transform(T1)
    T2 = sitk.Transform(T2)
    T_all = sitk.CompositeTransform([T0, T1, T2])
    aug_image = sitk.Resample(original_image, reference_image, T_all,
                              interpolator, default_intensity_value)

    return aug_image


def similarity3D_parameter_space_random_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale, n=1):
    """
    Create a list representing a random (uniform) sampling of the 3D similarity transformation parameter space.
    As the SimpleITK rotation parameterization uses the vector portion of a versor we don't have an intuitive way of
    specifying rotations. We therefor use the ZYX Euler angle parameterization and convert to versor.
    Args:
        thetaX, thetaY, thetaZ: Ranges of Euler angle values to use, in radians.
        tx, ty, tz: Ranges of translation values to use in mm.
        scale: Range of scale values to use.
        n: Number of samples.
    Return:
        List of lists representing the parameter space sampling [vx,vy,vz,tx,ty,tz,s].
    """

    scale = scale/2
    rad = 2*np.pi/360
    theta_x_vals = np.random.normal(scale=thetaX/2 * rad, size=n)
    theta_y_vals = np.random.normal(scale=thetaY/2 * rad, size=n)
    theta_z_vals = np.random.normal(scale=thetaZ/2 * rad, size=n)
    tx_vals = (tx[1] - tx[0]) * np.random.random(n) + tx[0]
    ty_vals = (ty[1] - ty[0]) * np.random.random(n) + ty[0]
    tz_vals = (tz[1] - tz[0]) * np.random.random(n) + tz[0]
    s_vals = np.random.normal(loc=1.0, scale=scale, size=n)

    if n != 1:
        res = list(zip(theta_x_vals, theta_y_vals, theta_z_vals, tx_vals, ty_vals, tz_vals, s_vals))
        return [list(eul2quat(*(p[0:3]))) + list(p[3:7]) for p in res]
    else:
        rot = eul2quat(*([theta_x_vals, theta_y_vals, theta_z_vals]))
        return [rot[0], rot[1], rot[2], float(tx_vals), float(ty_vals), float(tz_vals), float(s_vals)]


def composeScaleSkew_parameter_space_random_sampling(s):
    """
    Create a list representing a random (uniform) sampling of the 3D Compose Scale Skew Transform parameter space.
    Args: scale_factors, skew, axis, angle, translation, rotation_center
        s = (sx, sy, sz): Array with the scale factors in the 3 dimensions.
        skew = (k0, k1, k2): Array with the skew constants in the 3 dimensions.
        tx, ty, tz: Ranges (in array) of translation values to use in mm for each direction.
        n: Number of samples.
    Return:
        List of lists representing the parameter space sampling (tx,ty,tz,s).
    """

    # Given a scale factor s, a random value is extracted from a normal distribution of mean 1.0 and standard
    # deviation s/2. This means that 95% of the  extracted values will belong to the interval [1-s, 1+s].
    sx = s[0]/2
    sy = s[1]/2
    sz = s[2]/2

    sx = np.random.normal(loc=1.0, scale=sx, size=1)
    sy = np.random.normal(loc=1.0, scale=sy, size=1)
    sz = np.random.normal(loc=1.0, scale=sz, size=1)

    # No translation
    tx = 0
    ty = 0
    tz = 0

    # No shear
    k0 = 0
    k1 = 0
    k2 = 0

    return [[float(sx), float(sy), float(sz)], [tx, ty, tz], [k0, k1, k2]]


def augment(im, tar, shape, patch_size=128):
    data = [im, tar]
    dimension = data[0].GetDimension()  # dimension = 3

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)  # [depth, height, width] = [0 0 0] for now
    for img in data:
        reference_physical_size = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx
                                   in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    reference_origin = data[0].GetOrigin()
    reference_direction = data[0].GetDirection()

    # For isotropic pixels, you can specify the image size for one of the axes and the others are determined by
    # this choice. Below we choose to set the x-axis to 144 and the spacing set accordingly.
    reference_size_x = shape[2]
    reference_spacing = [reference_physical_size[0] / (
            reference_size_x - 1)] * dimension  # [572/(144-1)]*3 = [4 4 4]
    # 4 mm spacing between voxels
    reference_size = [int(phys_sz / spc + 1) for phys_sz, spc in
                      zip(reference_physical_size, reference_spacing)]
    # [572/4 + 1, 572/4 + 1, 760/4 + 1] = [144, 144, 191]

    reference_image = sitk.Image(reference_size, data[0].GetPixelIDValue())  # sitk.Image([144, 144, 191], 8)
    # sitk.Image.GetPixelID(): get pixel type (e.g. sitkInt8)

    reference_image.SetOrigin(reference_origin)  # [0 0 0]
    reference_image.SetSpacing(reference_spacing)  # [4 4 4]
    reference_image.SetDirection(reference_direction)  # [1 0 0 0 1 0 0 0 1]

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates
    # as this takes into account size, spacing and direction cosines. For the vast majority of images the
    # direction cosines are the identity matrix, but when this isn't the case simply multiplying the central
    # index by the spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    T1 = sitk.Similarity3DTransform()   # similarity 3D transform with rotation as a versor, and isotropic scaling
    # around a fixed center with translation
    T2 = sitk.ComposeScaleSkewVersor3DTransform()   # transform applies a versor rotation and translation & scale/skew
    # to the space

    # Transform which maps from the reference image to the current image with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(data[0].GetDirection())

    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(data[0].TransformContinuousIndexToPhysicalPoint(np.array(data[0].GetSize()) / 2.0))
    centering_transform.SetOffset(
        np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))

    # Set the augmenting transform's center so that rotation is around the image center.
    T1.SetCenter(reference_center)
    T2.SetCenter(reference_center)

    # The parameters are scale (+-4%), rotation angle (+-5 degrees), x translation, y translation, z translation
    similarity3d_params = similarity3D_parameter_space_random_sampling(thetaX=5,
                                                                       thetaY=5,
                                                                       thetaZ=5,
                                                                       tx=(0, 0),
                                                                       ty=(0, 0),
                                                                       tz=(0, 0),
                                                                       scale=0.0)

    composeScaleSkew_params = composeScaleSkew_parameter_space_random_sampling((0.04, 0.04, 0.04))

    image = augment_images_spatial_composite(data[0], reference_image, centering_transform, T1, T2,
                                             similarity3d_params, composeScaleSkew_params,
                                             interpolator=sitk.sitkBSpline)
    image = sitk.GetArrayFromImage(image)

    target = augment_images_spatial_composite(data[1], reference_image, centering_transform, T1, T2,
                                              similarity3d_params, composeScaleSkew_params,
                                              interpolator=sitk.sitkBSpline)
    target = sitk.GetArrayFromImage(target)

    z = random.randint(0, image.shape[0] - patch_size)
    y = random.randint(0, image.shape[1] - patch_size)
    x = random.randint(0, image.shape[2] - patch_size)

    # noise = np.random.normal(loc=1.0, scale=0.1, size=np.asarray([PATCH_SIZE] * 3))

    # Transforming size (D, W, H) to (D, W, H, C) where C is the number of channels
    image = image[z:z + patch_size, y:y + patch_size, x:x + patch_size]
    image[image < 0] = 0
    #image = np.multiply(image, noise)

    # Transforming size (D, W, H) to (D, W, H, C) where C is the number of channels
    target = target[z:z + patch_size, y:y + patch_size, x:x + patch_size]
    target[target < 0] = 0

    return image, target
