from torch.utils.data._utils.collate import default_collate
from dataset.transforms import *
from dataset.random_erasing import RandomErasing


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img):
        img_group, label = img
        return [self.worker(img) for img in img_group], label


class SplitLabel(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img):
        img_group, label = img
        return self.worker(img_group), label



def train_augmentation(input_size, flip=True):
    if flip:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            GroupRandomHorizontalFlip(is_flow=False)])
    else:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            # GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip_sth()])


def get_augmentation(training, input_size=224, config=None):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 256 if input_size == 224 else input_size

    normalize = GroupNormalize(input_mean, input_std)
    #if 'something' in config.data.dataset:
    if 'SS' in config.data_set:
        if scale_size == 256:
            groupscale = GroupScale((256, 320))
        else:
            groupscale = GroupScale(int(scale_size))
    else:
        groupscale = GroupScale(int(scale_size))


    common = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize])

    if training:
        auto_transform = None
        erase_transform = None
        if config.aa: ###!!! ss for True, k400 for False
            print('***'*20)
            print('use random_augment!!!')
            auto_transform = create_random_augment(
                input_size=256, # scale_size
                auto_augment="rand-m7-n4-mstd0.5-inc1",
                interpolation="bicubic"
            )
        # if config.rand_erase:
        #     print('***'*20)
        #     print('use Random_Erasing!!!')
        #     erase_transform = RandomErasing(
        #         0.25,
        #         mode='pixel',
        #         max_count=1,
        #         num_splits=1,
        #         device="cpu",
        #     )           

        train_aug = train_augmentation(
            input_size,
            flip=False if 'SS' in config.data_set else True)

        unique = torchvision.transforms.Compose([
            groupscale,
            train_aug,
            GroupRandomGrayscale(p=0 if 'SS' in config.data_set else 0.2),
        ])

        if auto_transform is not None:
            print('=> ########## Using RandAugment!')
            unique = torchvision.transforms.Compose([
                SplitLabel(auto_transform), unique])

        if erase_transform is not None:
            print('=> ########## Using RandErasing!')
            return torchvision.transforms.Compose([
                unique, common, SplitLabel(erase_transform)
            ])
            
        return torchvision.transforms.Compose([unique, common])

    else:
        unique = torchvision.transforms.Compose([
            groupscale,
            GroupCenterCrop(input_size)])
        return torchvision.transforms.Compose([unique, common])






def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels = zip(*batch)
    # print(inputs, flush=True)
    # print(labels, flush=True)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    inputs, labels = (
        default_collate(inputs),
        default_collate(labels),
    )
    return inputs, labels