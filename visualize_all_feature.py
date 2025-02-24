import numpy as np
import argparse

def main(args):
    # Load the dictionary from the .npy file.
    data = np.load(args.npy_file, allow_pickle=True).item()
    
    # Print keys.
    print("Keys in the npy file:", data.keys())
    
    # Retrieve features.
    ft_features = data.get("ft_features")
    zt_features = data.get("zt_features")
    
    if ft_features is not None:
        print("ft_features shape:", ft_features.shape)
    else:
        print("ft_features not found.")
    
    if zt_features is not None:
        print("zt_features shape:", zt_features.shape)
    else:
        print("zt_features not found.")
    
    # Optionally, print the contents (or part of it)
    print("\nft_features content (first 2 elements):")
    print(ft_features[:2])
    print("\nzt_features content (first 2 elements):")
    print(zt_features[:2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Print content of npy feature file")
    parser.add_argument("--npy_file", type=str, required=True, help="Path to the .npy file")
    args = parser.parse_args()
    main(args)
