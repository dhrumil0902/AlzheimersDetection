#!/bin/bash

# this file moves the .set files into a single folder so they can be loaded
# into EEGLab in a single selection

# Create the destination directory if it doesn't exist
mkdir -p ../data/eeg_files

# Loop through each subdirectory and move the .set file
for dir in ../data/ds004504/sub-*; do
    if [[ -d "$dir" ]]; then
        # Get only the subdirectory name (e.g., sub-001, sub-002)
        sub_dir=$(basename "$dir")

        # Define the path to the .set file
        eeg_file="$dir/eeg/${sub_dir}_task-eyesclosed_eeg.set"
        
        # Check if the file exists before moving
        if [[ -f "$eeg_file" ]]; then
            mv "$eeg_file" ../data/eeg_files/
            echo "Moved $eeg_file to ../data/eeg_files/"
        else
            echo "File $eeg_file does not exist"
        fi
    fi
done