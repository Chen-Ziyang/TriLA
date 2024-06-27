import torch
import numpy as np
from dataloaders.normalize import normalize_image
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from PIL import Image
import torchvision.transforms as standard_transforms


def get_train_transform(patch_size=(256, 256)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
    #                                            p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_color_transforms():
    # Image appearance transformations
    color_input_transform = []

    color_input_transform += [standard_transforms.ColorJitter(0.8, 0.8, 0.8, 0.3)]
    color_input_transform += [standard_transforms.GaussianBlur(kernel_size=5)]
    color_input_transform += [standard_transforms.ToTensor()]
    color_input_transform = standard_transforms.Compose(color_input_transform)

    return color_input_transform


def collate_fn_color_transform(batch):
    images, labels, names = zip(*batch)
    ori_img = np.stack(images, 0)
    color_img = []
    for i in range(len(images)):
        image = Image.fromarray((images[i]).transpose(1, 2, 0).astype(np.uint8)).convert('RGB')
        tr_transforms = get_color_transforms()
        image_color = np.array(tr_transforms(image)).astype(np.float32)
        color_img.append(image_color)
    color_img = np.stack(color_img, 0)
    data_dict = {'ori': ori_img, 'color': color_img}
    return data_dict


def collate_fn_w_transform(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'mask': label, 'name': name}
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)
    try:
        data_dict['mask'] = to_one_hot_list(data_dict['mask'])
    except:
        data_dict['mask'] = None
    return data_dict


def collate_fn_wo_transform(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'mask': label, 'name': name}
    try:
        data_dict['mask'] = to_one_hot_list(data_dict['mask'])
    except:
        data_dict['mask'] = None
    return data_dict


def to_one_hot_list(mask_list):
    list = []
    for i in range(mask_list.shape[0]):
        mask = to_one_hot(mask_list[i].squeeze(0))
        list.append(mask)
    return np.stack(list, 0)


def to_one_hot(pre_mask, classes=2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [1, 0]
    mask[pre_mask == 2] = [1, 1]
    return mask.transpose(2, 0, 1)
