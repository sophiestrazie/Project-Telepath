import os
import numpy as np

def update_npy_dict(new_key, new_features, save_path):
    """
    Update or create a .npy dictionary file.
    
    If the file exists, load it, update the dictionary with the new key and array,
    and save it back. If it doesn't exist, create a new dictionary and save it.
    
    Parameters:
        new_key (str): The key to insert or update (e.g., "s01e01a")
        new_features (np.ndarray): The features to store
        save_path (str): Path to the .npy file
    """
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True).item()
        data[new_key] = new_features
        print(f"Updated existing file: added/overwritten key '{new_key}'")
    else:
        data = {new_key: new_features}
        print(f"Created new file with key '{new_key}'")
    
    np.save(save_path, data)


save_path = "./Visual/visual_npy.npy"
new_key = "s01e01a"
new_features = np.random.rand(10, 2048)  # Example feature array
update_npy_dict(new_key, new_features, save_path)

update_npy_dict("s01e01b", np.random.rand(10, 2048), save_path)

