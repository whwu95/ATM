import os
import torchvision
from torchvision import transforms

from dataset.sthvideo import Video_dataset as SSvVideoClsDataset
from dataset.sth import Video_dataset as SSinferDataset
from dataset.charades import Video_dataset as CharadesDataset
from dataset.kinetics import Video_dataset as KineticsFrameDataset
from dataset.Augmentation import get_augmentation

from dataset.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            data_path = '/bpfs/v2_mnt/VIS/test/k400/train_320_frames'
            anno_path = 'lists/k400/kinetics_rgb_train_se320.txt'
            transform = get_augmentation(True, input_size=args.input_size, config=args)
            random_shift = True
            num_sample = 2
        elif test_mode is True:
            data_path = '/bpfs/v2_mnt/VIS/test/k400/kinetics_400_val_320_opencv'
            mode = 'test'
            anno_path = 'lists/k400/kinetics_rgb_val_se320.txt'
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            num_sample = 1
        else:  
            data_path = '/bpfs/v2_mnt/VIS/test/k400/kinetics_400_val_320_opencv'
            mode = 'validation'
            anno_path = 'lists/k400/kinetics_rgb_val_se320.txt'
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            num_sample = 1

        if mode == 'test':
            test_crops = args.test_num_crop
            test_clips = args.test_num_segment
            dense_sample = False
            
            input_mean = [0.48145466, 0.4578275, 0.40821073]
            input_std = [0.26862954, 0.26130258, 0.27577711]
        
            
            # crop size
            input_size = args.input_size
            scale_size = 256 if input_size == 224 else input_size
        
            # control the spatial crop
            if test_crops == 1: # one crop
                cropping = torchvision.transforms.Compose([
                    GroupScale(scale_size),
                    GroupCenterCrop(input_size),
                ])
            elif test_crops == 3:  # do not flip, so only 3 crops (left right center)
                cropping = torchvision.transforms.Compose([
                    GroupFullResSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 5:  # do not flip, so only 5 crops
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 10:
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        )
                ])
            else:
                raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(test_crops))

        
            dataset = KineticsFrameDataset(   
                root_path = data_path, 
                list_file = anno_path, 
                labels_file = 'lists/kinetics_400_labels.csv',
                random_shift=False, num_segments=args.num_frames,
                image_tmpl='img_{:05d}.jpg',
                test_mode = True,
                transform=torchvision.transforms.Compose([
                    cropping,
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize(input_mean,input_std),
                ]),
                dense_sample = dense_sample,
                test_clips = test_clips,
            )
        
        else:      
            dataset = KineticsFrameDataset(
                root_path = data_path, 
                list_file = anno_path, 
                labels_file = 'lists/kinetics_400_labels.csv',
                num_segments=args.num_frames, image_tmpl='img_{:05d}.jpg',
                transform=transform,
                random_shift=random_shift, dense_sample=False, num_sample=num_sample)
        nb_classes = 400

    
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = 'lists/sthv2/train_rgb.txt' 
            transform = get_augmentation(True, input_size=args.input_size, config=args)
            random_shift = True
            num_sample = 2
        elif test_mode is True:
            mode = 'test'
            anno_path = 'lists/sthv2/val_rgb.txt'
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            num_sample = 1
        else:  
            mode = 'validation'
            anno_path = 'lists/sthv2/val_rgb.txt'
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            num_sample = 1

        if mode == 'test':
            test_crops = args.test_num_crop
            test_clips = args.test_num_segment
            dense_sample = False
            
            input_mean = [0.48145466, 0.4578275, 0.40821073]
            input_std = [0.26862954, 0.26130258, 0.27577711]
        
            # crop size
            input_size = args.input_size
            scale_size = (256, 320) if input_size == 224 else input_size
        
            # control the spatial crop
            if test_crops == 1: # one crop
                cropping = torchvision.transforms.Compose([
                    GroupScale(scale_size),
                    GroupCenterCrop(input_size),
                ])
            elif test_crops == 3:  # do not flip, so only 3 crops (left right center)
                cropping = torchvision.transforms.Compose([
                    GroupFullResSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 5:  # do not flip, so only 5 crops
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 10:
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        )
                ])
            else:
                raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(test_crops))

        
            dataset = SSinferDataset(   
                root_path = args.data_path, 
                list_file = anno_path, 
                labels_file = 'lists/sth_labels.csv',
                random_shift=False, num_segments=args.num_frames,
                image_tmpl='{:06d}.jpg',
                test_mode = True,
                transform=torchvision.transforms.Compose([
                    cropping,
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize(input_mean,input_std),
                ]),
                dense_sample = dense_sample,
                test_clips = test_clips,
            )
        
        else:      
            dataset = SSvVideoClsDataset(
                root_path = args.data_path, 
                list_file = anno_path, 
                labels_file = 'lists/sth_labels.csv',
                num_segments=args.num_frames, image_tmpl='{:06d}.jpg',
                transform=transform,
                random_shift=random_shift, dense_sample=False, num_sample=num_sample)
        nb_classes = 174
        
        
    elif args.data_set == 'SSV1':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = 'lists/sthv1/train_rgb.txt' 
            transform = get_augmentation(True, input_size=args.input_size, config=args)
            random_shift = True
            num_sample = 2
        elif test_mode is True:
            mode = 'test'
            anno_path = 'lists/sthv1/val_rgb.txt' 
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            num_sample = 1
        else:  
            mode = 'validation'
            anno_path = 'lists/sthv1/val_rgb.txt' 
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            num_sample = 1

        if mode == 'test':
            test_crops = args.test_num_crop #3
            test_clips = args.test_num_segment #4
            dense_sample = False
            
            input_mean = [0.48145466, 0.4578275, 0.40821073]
            input_std = [0.26862954, 0.26130258, 0.27577711]
        
            # crop size
            input_size = args.input_size
            scale_size = (256, 320) if input_size == 224 else input_size
        
            # control the spatial crop
            if test_crops == 1: # one crop
                cropping = torchvision.transforms.Compose([
                    GroupScale(scale_size),
                    GroupCenterCrop(input_size),
                ])
            elif test_crops == 3:  # do not flip, so only 3 crops (left right center)
                cropping = torchvision.transforms.Compose([
                    GroupFullResSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 5:  # do not flip, so only 5 crops
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 10:
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        )
                ])
            else:
                raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(test_crops))

        
            dataset = SSinferDataset(   
                root_path = args.data_path, 
                list_file = anno_path, 
                labels_file = 'lists/sth_labels.csv',
                random_shift=False, num_segments=args.num_frames,
                image_tmpl='{:05d}.jpg',
                test_mode = True,
                transform=torchvision.transforms.Compose([
                    cropping,
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize(input_mean,input_std),
                ]),
                dense_sample = dense_sample,
                test_clips = test_clips,
            )
        
        else:      
            dataset = SSvVideoClsDataset(
                root_path = args.data_path, 
                list_file = anno_path, 
                labels_file = 'lists/sth_labels.csv',
                num_segments=args.num_frames, image_tmpl='{:05d}.jpg',
                transform=transform,
                random_shift=random_shift, dense_sample=False, num_sample=num_sample)
        nb_classes = 174

    elif args.data_set == 'Charades':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = 'lists/Charades/Charades_train_split_label.csv'
            transform = get_augmentation(True, input_size=args.input_size, config=args)
            random_shift = True
            test_mode = False
            num_sample = 2
        elif test_mode is True:
            mode = 'test'
            anno_path = 'lists/Charades/Charades_v1_test.csv'
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            test_mode = True
            num_sample = 1
        else:  
            mode = 'validation'
            anno_path = 'lists/Charades/Charades_v1_test.csv' 
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            test_mode = True
            num_sample = 1

        if mode == 'test':
            ###!!!
            test_crops = args.test_num_crop #3
            test_clips = args.test_num_segment #4
            dense_sample = False
            ###!!!
            
            input_mean = [0.48145466, 0.4578275, 0.40821073]
            input_std = [0.26862954, 0.26130258, 0.27577711]
        
            # crop size
            input_size = args.input_size
            scale_size = 256 if input_size == 224 else input_size
        
            # control the spatial crop
            if test_crops == 1: # one crop
                cropping = torchvision.transforms.Compose([
                    GroupScale(scale_size),
                    GroupCenterCrop(input_size),
                ])
            elif test_crops == 3:  # do not flip, so only 3 crops (left right center)
                cropping = torchvision.transforms.Compose([
                    GroupFullResSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 5:  # do not flip, so only 5 crops
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 10:
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        )
                ])
            else:
                raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(test_crops))

        
            dataset = CharadesDataset(
                root_path = args.data_path, 
                list_file = anno_path,
                labels_file = 'lists/Charades/Charades_v1_classes.txt',
                random_shift=random_shift, num_segments=args.num_frames,
                modality='RGB',
                image_tmpl='{}-{:06d}.jpg',
                test_mode=test_mode,
                transform=torchvision.transforms.Compose([
                    cropping,
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize(input_mean, input_std),
                ]),
                dense_sample=False,
                test_clips=test_clips,
                mode=mode)
        
        else:      
            dataset = CharadesDataset(
                root_path = args.data_path, 
                list_file = anno_path,
                labels_file = 'lists/Charades/Charades_v1_classes.txt',
                num_segments=args.num_frames,
                modality='RGB', image_tmpl='{}-{:06d}.jpg',
                random_shift=random_shift,
                transform=transform, 
                num_sample=num_sample, #dense_sample=config.data.dense,
                test_mode=test_mode,
                fps=24,
                mode=mode)
   
        nb_classes = 157

    elif args.data_set == 'anet':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            data_path = '/bpfs/v2_mnt/VIS/wuwenhao/anet/anet_instance_frames_v1.3_train_vids_fps1'
            anno_path = 'lists/anet/anet_train_instance_fps1.txt'
            transform = get_augmentation(True, input_size=args.input_size, config=args)
            random_shift = True
            num_sample = 2
        elif test_mode is True:
            data_path = '/bpfs/v2_mnt/VIS/wuwenhao/anet/activitynet_val_resize_img_256_340_fps1'
            mode = 'test'
            anno_path = 'lists/anet/anet_val_video_fps1.txt'
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            num_sample = 1
        else:  
            data_path = '/bpfs/v2_mnt/VIS/wuwenhao/anet/activitynet_val_resize_img_256_340_fps1'
            mode = 'validation'
            anno_path = 'lists/anet/anet_val_video_fps1.txt'
            transform = get_augmentation(False, input_size=args.input_size, config=args)
            random_shift = False
            num_sample = 1

        if mode == 'test':
            ###!!!
            test_crops = args.test_num_crop #3
            test_clips = args.test_num_segment #4
            dense_sample = False
            ###!!!
            
            input_mean = [0.48145466, 0.4578275, 0.40821073]
            input_std = [0.26862954, 0.26130258, 0.27577711]
        
            # crop size
            input_size = args.input_size
            scale_size = 256 if input_size == 224 else input_size
        
            # control the spatial crop
            if test_crops == 1: # one crop
                cropping = torchvision.transforms.Compose([
                    GroupScale(scale_size),
                    GroupCenterCrop(input_size),
                ])
            elif test_crops == 3:  # do not flip, so only 3 crops (left right center)
                cropping = torchvision.transforms.Compose([
                    GroupFullResSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 5:  # do not flip, so only 5 crops
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        flip=False)
                ])
            elif test_crops == 10:
                cropping = torchvision.transforms.Compose([
                    GroupOverSample(
                        crop_size=input_size,
                        scale_size=scale_size,
                        )
                ])
            else:
                raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(test_crops))

        
            dataset = KineticsFrameDataset(   
                root_path = data_path, 
                list_file = anno_path, 
                labels_file = 'lists/anet1.3_labels.csv',
                random_shift=False, num_segments=args.num_frames,
                image_tmpl='image_{:06d}.jpg',
                test_mode = True,
                transform=torchvision.transforms.Compose([
                    cropping,
                    Stack(roll=False),
                    ToTorchFormatTensor(div=True),
                    GroupNormalize(input_mean,input_std),
                ]),
                dense_sample = dense_sample,
                test_clips = test_clips,
            )
        
        else:      
            dataset = KineticsFrameDataset(
                root_path = data_path, 
                list_file = anno_path, 
                labels_file = 'lists/anet1.3_labels.csv',
                num_segments=args.num_frames, image_tmpl='image_{:06d}.jpg',
                transform=transform,
                random_shift=random_shift, dense_sample=False, num_sample=num_sample)
        nb_classes = 200

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
