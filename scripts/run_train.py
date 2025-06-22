import torch
import os
from telepath.fmriTransformer.model import FMRITransformerModel, Config
from telepath.fmriTransformer.train import train_one_epoch, evaluate
from telepath.fmriTransformer.data_load_align import prepare_and_save_aligned_data

# --- Parameters ---
root_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
subject = 1
train_episodes = ["s01e01a", "s01e02a", "s01e03a"]
val_episodes = ["s01e04a"]
save_train_path = "data/aligned_train.pkl"
save_val_path = "data/aligned_val.pkl"

# --- Prepare DataLoaders ---
train_loader = prepare_and_save_aligned_data(
    root_data_dir=root_data_dir,
    subject_id=subject,
    selected_episodes=train_episodes,
    window_size=10,
    stride=5,
    batch_size=4,
    save_path=save_train_path
)

val_loader = prepare_and_save_aligned_data(
    root_data_dir=root_data_dir,
    subject_id=subject,
    selected_episodes=val_episodes,
    window_size=10,
    stride=5,
    batch_size=4,
    save_path=save_val_path,
    shuffle=False
)

# --- Initialize Model ---
stim, fmri, _, _ = next(iter(train_loader))  # infer dimensions from one batch
config = Config(input_dim=stim.shape[-1], output_dim=fmri.shape[-1], )
model = FMRITransformerModel(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# --- Training Loop ---
for epoch in range(config.num_epochs):
    train_one_epoch(model, train_loader, optimizer, config.device, epoch)
    evaluate(model, val_loader, config.device, epoch)
