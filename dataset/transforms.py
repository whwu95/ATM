import torchvision
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import numbers
import math
import torch
from .rand_augment import rand_augment_transform

class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):

        img_group, label = img
        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images,label

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img):
        img_group, label = img
        return [self.worker(img) for img in img_group], label
    

class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img):
        img_group, label = img
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret,label
        else:
            return img_group, label

    
class GroupRandomHorizontalFlip_sth(object):
	"""Randomly horizontally flips the given PIL.Image with a probability of 0.5
	"""
	def __init__(self):
		self.target_transform = {86:87,87:86,93:94,94:93,166:167,167:166}

	def __call__(self, img):
		img_group, label = img
		v = random.random()
		if v < 0.5:
			ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
			if self.target_transform is not None:
				if label in self.target_transform:
					label = self.target_transform[label]
			return ret,label
		else:
			return img_group, label


class GroupNormalizeTV(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.worker = torchvision.transforms.Normalize(mean,std)

    def __call__(self, img):
        img_group, label = img
        return [self.worker(img) for img in img_group], label
    
        
class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, Te):
        tensor, label = Te
        mean = self.mean * (tensor.size()[0]//len(self.mean))
        std = self.std * (tensor.size()[0]//len(self.std))
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        if len(tensor.size()) == 3:
            # for 3-D tensor (T*C, H, W)
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif len(tensor.size()) == 4:
            # for 4-D tensor (C, T, H, W)
            tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor, label


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img):
        img_group, label = img
        return [self.worker(img) for img in img_group],label


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img):
        img_group, label = img
        if self.scale_worker is not None:
            img_group ,label = self.scale_worker(img)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)
        return oversample_group,label


class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img):
        img_group, label = img
        if self.scale_worker is not None:
            img_group ,label = self.scale_worker(img)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group,label



class GroupFCSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img):
        img_group, label = img

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fc_fix_offset(image_w, image_h, image_h, image_h)
        oversample_group = list()

        for o_w, o_h in offsets:
            normal_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + image_h, o_h + image_h))
                normal_group.append(crop)
            oversample_group.extend(normal_group)
        return oversample_group, label


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        img_group, label = img
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group, label

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]

        candidate_sizes = []

        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_distort:
                    candidate_sizes.append([w, h])

        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]
        crop_w, crop_h = crop_size

        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_w)
            h_offset = random.randint(0, image_h - crop_h)
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_w, crop_h)

        return crop_w, crop_h, w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        candidate_offsets = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step),  # center
        ]

        if more_fix_crop:
            extra_candidate_offsets = [
                (0, 2 * h_step),  # center left
                (4 * w_step, 2 * h_step),  # center right
                (2 * w_step, 4 * h_step),  # lower center
                (2 * w_step, 0 * h_step),  # upper center
                (1 * w_step, 1 * h_step),  # upper left quarter
                (3 * w_step, 1 * h_step),  # upper right quarter
                (1 * w_step, 3 * h_step),  # lower left quarter
                (3 * w_step, 3 * h_step)  # lower right quarter
            ]
            candidate_offsets.extend(extra_candidate_offsets)
        return candidate_offsets


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img_group, label = img
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group, label
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img):
        img_group, label = img
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2),label
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2),label

class Stack1(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img):
        img_group, label = img
        if self.roll:
            return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label
        else:
            
            rst = np.concatenate(img_group, axis=0)
            # plt.imshow(rst[:,:,3:6])
            # plt.show()
            return torch.from_numpy(rst), label

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pi):
        pic,label = pi
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float(),label
    

class ToTorchFormatTensor1(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.worker = torchvision.transforms.ToTensor()
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class IdentityTransform(object):

    def __call__(self, data):
        return data
    
# custom transforms
class GroupRandomColorJitter(object):
    """Randomly ColorJitter the given PIL.Image with a probability
    """
    def __init__(self, p=0.8, brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1):
        self.p = p
        self.worker = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                        saturation=saturation, hue=hue)

    def __call__(self, img):
        img_group, label = img
        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]
            
            return ret, label
        else:
            return img_group, label

class GroupRandomGrayscale(object):
    """Randomly Grayscale flips the given PIL.Image with a probability
    """
    def __init__(self, p=0.2):
        self.p = p
        self.worker = torchvision.transforms.Grayscale(num_output_channels=3)

    def __call__(self, img):
        img_group, label = img
        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]
            
            return ret, label
        else:
            return img_group, label

class GroupGaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        img_group, label = img
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return [img.filter(ImageFilter.GaussianBlur(sigma))  for img in img_group], label
        else:
            return img_group, label

class GroupSolarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        img_group, label = img
        if random.random() < self.p:
            return [ImageOps.solarize(img)  for img in img_group], label
        else:
            return img_group, label


def _pil_interp(method):
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR

def create_random_augment(
    input_size,
    auto_augment=None,
    interpolation="bilinear",
):
    """
    Get video randaug transform.

    Args:
        input_size: The size of the input video in tuple.
        auto_augment: Parameters for randaug. An example:
            "rand-m7-n4-mstd0.5-inc1" (m is the magnitude and n is the number
            of operations to apply).
        interpolation: Interpolation method.
    """
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = {"translate_const": int(img_size_min * 0.45)}
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            return torchvision.transforms.Compose(
                [rand_augment_transform(auto_augment, aa_params)]
            )
    raise NotImplementedError