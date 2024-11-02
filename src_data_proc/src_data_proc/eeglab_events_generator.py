#this file generates an event file containing the five quartile start times for each EEGLab dataset

import os
import json
import math

# Base directory path
base_dir = "..\data\ds004504"

min_duration = math.inf

# Loop through each subdirectory from 001 to 088
for i in range(1, 89):
    # Format subdirectory and JSON file paths
    sub_id = f"{i:03}"  # Pads i to 3 digits, e.g., 001, 002, etc.
    sub_dir = os.path.join(base_dir, f"sub-{sub_id}\eeg")
    json_file = os.path.join(sub_dir, f"sub-{sub_id}_task-eyesclosed_eeg.json")
    
    try:
        # Check if JSON file exists in the subdirectory
        if os.path.isfile(json_file):
            # Open and load JSON data
            with open(json_file, 'r') as file:
                data = json.load(file)
                
                # Read the RecordingDuration field
                recording_duration = data.get("RecordingDuration")
                
                if recording_duration < min_duration:
                    min_duration = recording_duration
                
                c1 = 10
                c5 = recording_duration - 10
                dc = (c5 - c1) / 4

                c_curr = c1
                lines = ["type\tlatency\n"]
                for j in range(5):
                    line = "nah\t" + str(c_curr - 6) + "\n"
                    lines.append(line)
                    c_curr += dc

                with open(f"../data/eeg_events/sub-{sub_id}_events_5_12.txt", 'w') as file:
                    file.writelines(lines)
        else:
            print(f"sub-{sub_id}: JSON file not found.")
    
    except json.JSONDecodeError:
        print(f"sub-{sub_id}: Failed to decode JSON.")
    except Exception as e:
        print(f"sub-{sub_id}: Error - {e}")

print("min_duration: ", min_duration)