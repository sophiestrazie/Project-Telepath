import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
import wandb


def compute_encoding_accuracy(fmri_val, fmri_val_pred):
    """
    Computes voxel-wise Pearson correlation and returns mean correlation.
    fmri_val, fmri_val_pred: [T, D]
    """
    fmri_val = fmri_val.detach().cpu().numpy()
    fmri_val_pred = fmri_val_pred.detach().cpu().numpy()
    scores = np.zeros(fmri_val.shape[1], dtype=np.float32)
    for i in range(fmri_val.shape[1]):
        scores[i] = pearsonr(fmri_val[:, i], fmri_val_pred[:, i])[0]
    return scores.mean()


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_pearson = 0
    num_batches = 0

    for stim, fmri in tqdm(dataloader, desc=f"Epoch {epoch+1} - Training"):
        stim, fmri = stim.to(device), fmri.to(device)

        optimizer.zero_grad()
        output = model(input_seq=stim, fmri_seq=fmri)

        loss = nn.functional.mse_loss(output, fmri)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Pearson correlation
        batch_size = stim.size(0)
        for i in range(batch_size):
            pred = output[i]
            target = fmri[i]
            total_pearson += compute_encoding_accuracy(target, pred)
            num_batches += 1

    avg_loss = total_loss / len(dataloader)
    avg_pearson = total_pearson / num_batches

    print(f"[Train] MSE: {avg_loss:.4f}, Pearson: {avg_pearson:.4f}")
    wandb.log({
        "train/mse": avg_loss,
        "train/pearson": avg_pearson,
        "epoch": epoch
    })

    return avg_loss, avg_pearson


@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()
    total_pearson = 0
    total_raw_mse = 0
    num_batches = 0

    for stim, fmri in tqdm(dataloader, desc=f"Epoch {epoch+1} - Evaluating"):
        stim, fmri = stim.to(device), fmri.to(device)
        batch_size = stim.shape[0]

        for i in range(batch_size):
            seq_len = stim[i].shape[0]
            pred = model.autoregressive_inference(
                input_seq=stim[i:i+1],
                seq_len=seq_len,
                start_token=fmri[i, 0]
            )
            pred = pred.squeeze(0)
            truth = fmri[i]

            total_pearson += compute_encoding_accuracy(truth, pred)
            total_raw_mse += nn.functional.mse_loss(pred, truth).item()
            num_batches += 1

    mean_pearson = total_pearson / num_batches
    mean_raw_mse = total_raw_mse / num_batches

    print(f"[Eval] Pearson: {mean_pearson:.4f}, Raw MSE: {mean_raw_mse:.4f}")
    wandb.log({
        "eval/pearson": mean_pearson,
        "eval/raw_mse": mean_raw_mse,
        "epoch": epoch
    })

    return mean_pearson, mean_raw_mse
