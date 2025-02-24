import numpy as np

# Load the features from the file
features = np.load("/nas/spatial/source_datasets/scannet/datasets/scans_videos/scene0192_01.mp4.cut3r.npy")

# Check the shape and type of the array
print("Shape:", features.shape)
print("Data type:", features.dtype)

# Optionally, view a subset of the features
print("First few features:", features[:5])