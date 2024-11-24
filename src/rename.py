import os
import glob

# Directory containing the files
directory = "data/"

# Pattern to match files
pattern = os.path.join(directory, "epoch_*_*.npy")

# Find and rename matching files
for file_path in glob.glob(pattern):
    # Extract the base name of the file
    base_name = os.path.basename(file_path)  # e.g., "epoch_123_456.npy"
    
    # Replace "epoch" with "cfc" in the base name
    new_base_name = base_name.replace("epoch", "cfc", 1)  # e.g., "cfc_123_456.npy"
    
    # Create the full path for the new file name
    new_file_path = os.path.join(directory, new_base_name)
    
    # Rename the file
    os.rename(file_path, new_file_path)
    print(f"Renamed: {file_path} -> {new_file_path}")
