import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
from telepath.fmriTransformer.model import autoregressive_inference

def masked_mse_loss(predictions, targets, mask):
    """
    predictions: [B, T, D]
    targets: [B, T, D]
    mask: [B, T] (True = pad)
    """
    mask = ~mask.unsqueeze(-1)  # [B, T, 1] — True where valid
    loss = (predictions - targets) ** 2
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)

def raw_mse(predictions, targets):
    return ((predictions - targets) ** 2).mean().item()


def compute_encoding_accuracy(fmri_val, fmri_val_pred):
    """
    Computes voxel-wise Pearson correlation and returns mean correlation.
    fmri_val, fmri_val_pred: [T, D]
    """
    fmri_val = fmri_val.cpu().numpy()
    fmri_val_pred = fmri_val_pred.cpu().numpy()
    scores = np.zeros(fmri_val.shape[1], dtype=np.float32)
    for i in range(fmri_val.shape[1]):
        scores[i] = pearsonr(fmri_val[:, i], fmri_val_pred[:, i])[0]
    return scores.mean()


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_raw_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} - Training"):
        stim, fmri, src_mask, tgt_mask = [x.to(device) for x in batch]

        optimizer.zero_grad()
        output = model(
            input_seq=stim,
            fmri_seq=fmri,
            src_padding_mask=src_mask,
            tgt_padding_mask=tgt_mask
        )

        loss = masked_mse_loss(output, fmri, tgt_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_raw_loss += raw_mse(output, fmri)

    avg_loss = total_loss / len(dataloader)
    avg_raw = total_raw_loss / len(dataloader)
    print(f"[Train] Masked MSE: {avg_loss:.4f}, Raw MSE: {avg_raw:.4f}")
    return avg_loss, avg_raw


@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()
    total_pearson = 0
    total_raw_mse = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1} - Evaluating"):
        stim, fmri, src_mask, tgt_mask = [x.to(device) for x in batch]

        batch_size = stim.shape[0]
        for i in range(batch_size):
            seq_len = stim[i].shape[0]
            pred = model.autoregressive_inference(
                input_seq=stim[i:i+1],
                seq_len=seq_len,
                start_token=fmri[i, 0]  # start token = first fmri
            )  # [1, T, D]
            pred = pred.squeeze(0)
            truth = fmri[i]

            total_pearson += compute_encoding_accuracy(truth, pred)
            total_raw_mse += raw_mse(pred, truth)
            num_batches += 1

    mean_pearson = total_pearson / num_batches
    mean_raw_mse = total_raw_mse / num_batches
    print(f"[Eval] Pearson: {mean_pearson:.4f}, Raw MSE: {mean_raw_mse:.4f}")
    return mean_pearson, mean_raw_mse
