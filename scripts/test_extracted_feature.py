import h5py

file_path = r'extracted_features\audio\s01e01a_audio_features.h5'


with h5py.File(file_path, 'r') as f:
    # List all groups
    print("Groups:")
    for group in f:
        print(f"  {group}")

    # Dive into a group
    group_name = 's01e01a'  # Replace with actual group name
    if group_name in f:
        print("\nDatasets in group:")
        for key in f[group_name]:
            print(f"  {key} -> shape: {f[group_name][key].shape}")
        
        # Access data
        audio_data = f[group_name]['audio'][:]
        print("\nAudio feature shape:", audio_data.shape)
