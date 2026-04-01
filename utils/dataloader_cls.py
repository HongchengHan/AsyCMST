import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from glob2 import glob
import os
from PIL import Image
from easydict import EasyDict as edict
import yaml
import pandas as pd
import numpy as np
from torchvision.utils import save_image
from pytorchvideo.data.encoded_video import EncodedVideo
import random


class BUS_CEUS_Classification(Dataset):
    def __init__(self, root_dir, label_csv, video_size=224, 
                 num_frm=16, start_sec=0.0, end_sec=10.0, 
                 mode='train', augment=False, aug=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.video_size = (video_size, video_size)
        self.num_frm = num_frm
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.mode = mode
        self.augment = augment
        self.aug = aug
        self.ids = np.array(pd.read_csv(os.path.join(root_dir, label_csv))['id'])
        self.bus_ids = np.array(pd.read_csv(os.path.join(root_dir, label_csv))['bus_vid'])
        self.ceus_ids = np.array(pd.read_csv(os.path.join(root_dir, label_csv))['ceus_vid'])
        self.labels = np.array(pd.read_csv(os.path.join(root_dir, label_csv))['label'])
        self.category = {'0': '0.ben', '1':'1.mal'}
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def get_video(self, vid_path):
        vid = EncodedVideo.from_path(vid_path)
        vid_data = vid.get_clip(self.start_sec, self.end_sec)
        vid_data = vid_data['video'] / 255
        return vid_data

    def uniform_temporal_subsample(self, x: torch.Tensor, num_samples: int, temporal_dim: int = -3) -> torch.Tensor:
        t = x.shape[temporal_dim]
        assert num_samples > 0 and t > 0
        # Sample by nearest neighbor interpolation if num_samples > t.
        indices = torch.linspace(0, t - 1, num_samples)
        indices = torch.clamp(indices, 0, t - 1).long()
        return torch.index_select(x, temporal_dim, indices)
    
    def spatial_transform(self):
        s_trans = transforms.Compose([
            # transforms.Resize(self.video_size, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(self.video_size, scale=(0.8,1.0),ratio=(0.80, 1.25), antialias=True),
            transforms.RandomRotation(degrees=(-45,45)),
            # transforms.ColorJitter(brightness=(0,0.1), contrast=(0,0.1))
            # transforms.ColorJitter(brightness=(0,0.2),contrast=(0,0.2),saturation=(0,0.2),hue=(0,0.2)),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.ToTensor()
        ])
        return s_trans
    
    def temporal_transform(self):
        t_trans = self.uniform_temporal_subsample
        return t_trans
    
    def preprocess_video(self, vid_tensor):       
        s_trans, t_trans = self.spatial_transform(), self.temporal_transform()    
        segment = t_trans(vid_tensor, self.num_frm)
        segment = segment.permute(1, 0, 2, 3)
        segment = s_trans(segment)
        segment = segment.permute(1, 0, 2, 3)
        return segment

    def test(self):
        print(self.category)


    def __getitem__(self, index) -> dict:
        bus_path = os.path.join(self.root_dir, 'videos', self.category[str(self.labels[index])], 'bus', self.bus_ids[index])
        ceus_path = os.path.join(self.root_dir, 'videos', self.category[str(self.labels[index])], 'ceus', self.ceus_ids[index])
        vid_bus = self.get_video(bus_path)
        vid_ceus = self.get_video(ceus_path)
        # print(vid_face.size())
        segment_bus = self.preprocess_video(vid_bus)
        segment_ceus = self.preprocess_video(vid_ceus)
        bn_label = np.array(self.labels[index])
        bn_label = torch.from_numpy(bn_label).type(torch.LongTensor)
        # sample_id = np.array(self.ids[index])
        sample_id = np.array([0])
        
        return {'segment_bus': segment_bus, 
                'segment_ceus': segment_ceus, 
                'label': bn_label}



class BUS_CEUS_Classification_Images(Dataset):
    def __init__(self, root_dir, label_csv, video_size=224, 
                 num_frm=16, start_sec=0.0, end_sec=10.0, 
                 mode='train', augment=False, aug=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.video_size = (video_size, video_size)
        self.num_frm = num_frm
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.mode = mode
        self.augment = augment
        self.aug = aug
        self.ids = np.array(pd.read_csv(os.path.join(root_dir, label_csv))['id'])
        self.bus_ids = np.array(pd.read_csv(os.path.join(root_dir, label_csv))['bus_vid'])
        self.ceus_ids = np.array(pd.read_csv(os.path.join(root_dir, label_csv))['ceus_vid'])
        self.labels = np.array(pd.read_csv(os.path.join(root_dir, label_csv))['label'])
        self.category = {'0': '0.ben', '1':'1.mal'}
        
    def __len__(self) -> int:
        return len(self.ids)
    
    
    def get_images(self, image_dir, roi_box=None):
        image_path_list = glob(os.path.join(image_dir, '*'))
        selected_indexes = np.linspace(0, len(image_path_list)-1, self.num_frm).astype(int).tolist()
        if not roi_box is None:
            # image_list = [img.crop(roi_box) for img in image_list]
            image_list = [Image.open(image_path_list[idx]).crop(roi_box) for idx in selected_indexes]
        else:
            image_list = [Image.open(image_path_list[idx]) for idx in selected_indexes]
        
        return image_list
    
    # def spatial_transform(self):
    #     s_trans = transforms.Compose([
    #         # transforms.Resize(self.video_size, antialias=True),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.RandomResizedCrop(self.video_size, scale=(0.8,1.0),ratio=(0.80, 1.25), antialias=True),
    #         transforms.RandomRotation(degrees=(-45,45)),
    #         transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1,hue=0.1),
    #         transforms.ToTensor(),
    #         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ])
    #     return s_trans
    
    def preprocess_video(self, bus_image_list, ceus_image_list):
        # s_trans = self.spatial_transform()
        transform_list = []
        
        if self.mode == 'train':
            # seed = random.randint(0, 100)
            torch.random.seed()
            
            p = random.randint(0, 1)
            transform_list.append(transforms.RandomHorizontalFlip(p))

            p = random.randint(0, 1)
            transform_list.append(transforms.RandomHorizontalFlip(p))
            
            scale_param, ratio_param = random.uniform(0.8, 1.0), random.uniform(0.8, 1.25)
            transform_list.append(transforms.RandomResizedCrop(self.video_size, scale=(scale_param,scale_param),ratio=(ratio_param, ratio_param), antialias=True))

            angle = random.uniform(-45, 45)
            transform_list.append(transforms.RandomRotation(degrees=(angle,angle)))
            
            # transform_list.append(transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1,hue=0.1))
        
        
        transform_list.append(transforms.Resize(self.video_size, antialias=True))
        transform_list.append(transforms.ToTensor())
        
        s_trans = transforms.Compose(transform_list)
        
        bus_augmented_frames = [s_trans(img) for img in bus_image_list]
        bus_augmented_video = torch.stack([frame for frame in bus_augmented_frames]) # T, C, H, W
        bus_segment = bus_augmented_video.permute(1, 0, 2, 3) # C, T, H, W
        
        ceus_augmented_frames = [s_trans(img) for img in ceus_image_list]
        ceus_augmented_video = torch.stack([frame for frame in ceus_augmented_frames]) # T, C, H, W
        ceus_segment = ceus_augmented_video.permute(1, 0, 2, 3) # C, T, H, W
        
        return bus_segment, ceus_segment

    def test(self):
        print(self.category)


    def __getitem__(self, index) -> dict:
        bus_path = os.path.join(self.root_dir, 'images', self.category[str(self.labels[index])], 'bus', self.bus_ids[index][:-4])
        ceus_path = os.path.join(self.root_dir, 'images', self.category[str(self.labels[index])], 'ceus', self.ceus_ids[index][:-4])
        vid_bus = self.get_images(bus_path)
        vid_ceus = self.get_images(ceus_path)
        segment_bus, segment_ceus = self.preprocess_video(vid_bus, vid_ceus)
        bn_label = np.array(self.labels[index])
        bn_label = torch.from_numpy(bn_label).type(torch.LongTensor)
        
        return {'segment_bus': segment_bus, 
                'segment_ceus': segment_ceus, 
                'label': bn_label}


if __name__ == '__main__':
    config_path = '/data/hanhc/Ultrasound/AsyCMST/config/train_asycmst_bus_ceus.yaml'
    config = edict(yaml.load(open(config_path), Loader=yaml.FullLoader))
    # train_dir = os.path.join(config.data.root_dir, 'train')
    root_dir = '/data/hanhc/Ultrasound/AsyCMST/datasets/XJTU-MMUS-subset-20260401'
    train_label = 'train_1.csv'
    
    train_set = BUS_CEUS_Classification_Images(
        root_dir, 
        train_label, 
        config.data.size, 
        num_frm=16,
        mode='train', 
        augment=config.data.augment, 
        aug=config.data.aug)
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, 
        num_workers=16, pin_memory=True, drop_last=False)
    print(train_set.ids)
    print(len(train_set))
    i = 0
    for batch in train_loader:
        print(f'batch{i}')
        vid_bus, vid_ceus, label = batch['segment_bus'], batch['segment_ceus'], ['label']
        save_image(vid_bus[0, :, 8, :, :].float(), f'/data/hanhc/Ultrasound/AsyCMST/demo_image/{i}_bus.jpg')
        save_image(vid_ceus[0, :, 8, :, :].float(), f'/data/hanhc/Ultrasound/AsyCMST/demo_image/{i}_ceus.jpg')
        print(vid_bus.size(), vid_ceus.size(), torch.max(vid_bus[0, :, 8, :, :]), torch.max(vid_ceus[0, :, 8, :, :]))
        i += 1
    
   