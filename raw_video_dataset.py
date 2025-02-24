import os
import os.path as osp
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RawVideoDataset(Dataset):
    """
    A dataset that reads raw video files from a given directory.
    It expects that the directory contains video files with names following the
    pattern {video_id}.mp4. For each video file, it extracts a fixed number of
    evenly spaced frames and returns a list of dictionaries (one per frame).

    Dummy keys ('img_mask', 'ray_mask', 'ray_map', 'reset', 'update') are added to satisfy model requirements.
    """
    def __init__(self, video_dir, num_frames=32, transform=None):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.samples = []
        # Look for .mp4 files directly in the given directory.
        for filename in os.listdir(video_dir):
            if filename.lower().endswith('.mp4'):
                video_file = osp.join(video_dir, filename)
                video_id = osp.splitext(filename)[0]
                self.samples.append({
                    'video_path': video_file,
                    'video_id': video_id,
                    'video_folder': video_dir
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        video_id = sample['video_id']
        video_folder = sample['video_folder']

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Compute indices for exactly num_frames evenly spaced frames.
        if total_frames < self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / float(self.num_frames)
            frame_indices = [int(step * i) for i in range(self.num_frames)]
        
        frames_list = []
        current_frame = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in frame_indices:
                # Convert BGR (OpenCV default) to RGB.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_tensor = self.transform(frame)  # shape: [3, 224, 224]
                image_tensor = image_tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]
                true_shape = torch.tensor([224, 224]).unsqueeze(0)
                # Create a dummy ray_map with shape (1, 224, 224, 6)
                dummy_ray_map = torch.zeros(1, 224, 224, 6)
                frames_list.append({
                    'img': image_tensor,
                    'true_shape': true_shape,
                    'label': f"{video_id}.mp4",
                    'video_dir': video_folder,
                    'video_id': video_id,
                    'img_mask': torch.ones(1, dtype=torch.bool),
                    'ray_mask': torch.ones(1, dtype=torch.bool),
                    'ray_map': dummy_ray_map,
                    'reset': torch.zeros(1, dtype=torch.bool),
                    'update': torch.ones(1, dtype=torch.bool)
                })
            current_frame += 1
        cap.release()
        return frames_list
