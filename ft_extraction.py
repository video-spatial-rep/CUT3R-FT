import os
import os.path as osp
import numpy as np
import argparse

def process_file(file_path):
    print("Processing file:", file_path)
    # Load the .npy file (allow_pickle in case it's a dictionary)
    loaded = np.load(file_path, allow_pickle=True)
    
    # If loaded is an array of dtype object, try to extract its dictionary content
    if isinstance(loaded, np.ndarray) and loaded.dtype == np.object_:
        data = loaded.item()
    else:
        data = loaded

    # If data is a dict, extract ft_features; otherwise assume data is already the feature array.
    if isinstance(data, dict):
        ft_features = data.get("ft_features")
        if ft_features is None:
            print(f"File {file_path} does not contain key 'ft_features'. Skipping.")
            return
    else:
        ft_features = data

    # Overwrite the file with only the ft_features NumPy array.
    np.save(file_path, ft_features)
    print(f"Saved only ft_features to {file_path}\n")

def main(args):
    target_dir = args.dir
    if not osp.isdir(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    # Get all files matching the pattern *.mp4.cut3r.npy in the target directory.
    npy_files = [osp.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".mp4.cut3r.npy")]
    
    if not npy_files:
        print("No matching files found in", target_dir)
        return

    print(f"Found {len(npy_files)} files in {target_dir}")
    for file_path in npy_files:
        process_file(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overwrite npy files with only the ft_features array (f feature)."
    )
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory containing .mp4.cut3r.npy files")
    args = parser.parse_args()
    main(args)
