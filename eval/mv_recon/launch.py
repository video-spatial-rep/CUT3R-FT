import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
import os.path as osp
from torchvision import transforms
from torch.utils.data import DataLoader
from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser("Generate Features from Raw Videos", add_help=True)
    parser.add_argument("--weights", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference")
    parser.add_argument("--model_name", type=str, default="ours", help="Model name (e.g. ours or cut3r)")
    parser.add_argument("--output_dir", type=str, default="features_output", help="Directory to save feature files")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory with raw video subfolders")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of frames to extract per video")
    return parser

def main(args):
    # Add the model path and setup accelerator.
    add_path_to_dust3r(args.weights)
    accelerator = Accelerator()
    device = accelerator.device

    # Load and set model to evaluation mode.
    if args.model_name in ["ours", "cut3r"]:
        from src.dust3r.model import ARCroco3DStereo
        model = ARCroco3DStereo.from_pretrained(args.weights).to(device)
        model.eval()
    else:
        raise NotImplementedError("Only 'ours' or 'cut3r' models are supported.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Import your RawVideoDataset. Ensure this class is defined in raw_video_dataset.py.
    from raw_video_dataset import RawVideoDataset

    # Define the transformation (match your dataset's expected size).
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create the dataset and DataLoader.
    video_dataset = RawVideoDataset(args.video_dir, num_frames=args.num_frames, transform=transform)
    # Use a collate function so that each sample is a list of frame dictionaries for one video.
    video_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    # Process each video to generate features.
    for video_frames in tqdm(video_loader, desc="Processing videos"):
        # Each item (video_frames) is a list of dictionaries (each representing one frame).
        # Move all tensor data to the specified device.
        for frame in video_frames:
            for key, value in frame.items():
                if isinstance(value, torch.Tensor):
                    frame[key] = value.to(device, non_blocking=True)

        with torch.no_grad():
            start = time.time()
            output = model(video_frames)
            end = time.time()
            # Extract enriched features from the model's output.
            ft_features = output.Ft_prime  # enriched image tokens
            zt_features = output.zt_prime   # enriched pose token

        fps = len(video_frames) / (end - start)
        print(f"Processed video {video_frames[0]['video_id']} at {fps:.2f} FPS")

        # Save the features to a .npy file.
        save_dict = {
            "ft_features": ft_features.cpu().numpy(),
            "zt_features": zt_features.cpu().numpy()
        }
        video_id = video_frames[0]['video_id']
        np.save(osp.join(args.output_dir, f"{video_id}.mp4.cut3r.npy"), save_dict)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
