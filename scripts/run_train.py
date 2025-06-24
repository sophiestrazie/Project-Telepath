import torch
import os
from telepath.fmriTransformer.model import FMRITransformerModel, Config
from telepath.fmriTransformer.train import train_one_epoch, evaluate
from telepath.fmriTransformer.data_load_align import prepare_and_save_aligned_data
from torch.utils.data import DataLoader
import wandb

# Configuration
root_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
subject = 1
modality = "all"
window_size = 10
stride = 5
batch_size = 128

# Train/val split
train_episodes = [f"s01e{episode:02d}{half}" for episode in range(1, 25) for half in ['a', 'b']]
val_episodes = [f"s02e{episode:02d}{half}" for episode in range(1, 13) for half in ['a', 'b']]

# Load data
train_loader = prepare_and_save_aligned_data(
    root_data_dir=root_data_dir,
    subject_id=subject,
    selected_episodes=train_episodes,
    window_size=window_size,
    stride=stride,
    batch_size=batch_size,
    excluded_trs_start=0,
    excluded_trs_end=0,
    hrf_delay=3,
    shuffle=True,
    save_path="/content/aligned/train.pkl"
)

val_loader = prepare_and_save_aligned_data(
    root_data_dir=root_data_dir,
    subject_id=subject,
    selected_episodes=val_episodes,
    window_size=window_size,
    stride=stride,
    batch_size=batch_size,
    excluded_trs_start=0,
    excluded_trs_end=0,
    hrf_delay=3,
    shuffle=False,
    save_path="/content/aligned/val.pkl"
)

# Get input/output dimensions from a sample batch
stim_batch, fmri_batch = next(iter(train_loader)) 
example_stim = stim_batch[0]
example_fmri = fmri_batch[0]



input_dim = example_stim.shape[-1]
output_dim = example_fmri.shape[-1]


# Initialize model
config = Config(input_dim=input_dim, output_dim=output_dim, window_size= window_size, stride=stride)
model = FMRITransformerModel(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# Initialize WandB
wandb.init(
    project="fmri-transformer",
    config=config.__dict__
)

# Training loop
for epoch in range(config.num_epochs):
    train_one_epoch(model, train_loader, optimizer, config.device, epoch)
    evaluate(model, val_loader, config.device, epoch)

    checkpoint_path = f"/content/checkpoints/epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # upload to WandB
    wandb.save(checkpoint_path)  