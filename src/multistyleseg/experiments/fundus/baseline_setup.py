from fundus_data_toolkit.data_aug import DAType
from multistyleseg.trainers.measure import LossType

def get_configurations():
    imgs_resolutions = [1536, 1024, 512, 256]
    data_augmentation = [DAType.HEAVY, DAType.MEDIUM, DAType.LIGHT, DAType.NONE]
    loss_types = [LossType.DICE_CE, LossType.FOCAL, LossType.CE, LossType.DICE]
    return imgs_resolutions, data_augmentation, loss_types

def iterable_configurations():
    imgs_resolutions, data_augmentation, loss_types = get_configurations()
    for img_res in imgs_resolutions:
        for da_type in data_augmentation:
            for loss_type in loss_types:
                yield img_res, da_type, loss_type