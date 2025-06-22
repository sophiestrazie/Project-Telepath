import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# hyperparameters for transformer 
class Config:
    def __init__(
        self,
        input_dim=1024,
        output_dim=1000,
        d_model=2000,
        nhead=4,
        nhid=128,
        nlayers=2,
        dropout=0.1,
        batch_size=4,
        lr=1e-3,
        num_epochs=10,
        device=None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

    # x: [batch, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].transpose(0, 1)  # [1, seq_len, d_model] to make sure batch first
        return self.dropout(x)
    

class FMRITransformerModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.output_projection = nn.Linear(config.output_dim, config.d_model)
        self.regressor = nn.Linear(config.d_model, config.output_dim)
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)

        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.nlayers,
            num_decoder_layers=config.nlayers,
            dim_feedforward=config.nhid,
            dropout=config.dropout,
            batch_first=True
        )

    def forward(
        self,
        input_seq,            # [B, T, input_dim]
        fmri_seq,             # [B, T, output_dim]
        src_lengths=None,     # optional, not used anymore
        tgt_lengths=None,     # optional, not used anymore
        src_padding_mask=None,  # [B, T]
        tgt_padding_mask=None   # [B, T]
    ):
        """
        input_seq: [B, T, input_dim]
        fmri_seq: [B, T, output_dim]
        src_padding_mask: [B, T] (True = pad)
        tgt_padding_mask: [B, T] (True = pad)
        """
        input_seq = self.pos_encoder(self.input_projection(input_seq))  # [B, T, d_model]
        fmri_seq = self.pos_encoder(self.output_projection(fmri_seq))   # [B, T, d_model]

        T = fmri_seq.size(1)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=input_seq.device), diagonal=1)

        out = self.transformer(
            src=input_seq,
            tgt=fmri_seq,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.regressor(out)  # [B, T, output_dim]

    @torch.no_grad()
    def autoregressive_inference(self, input_seq, seq_len, start_token):
        """
        input_seq: [1, T_in, input_dim]
        start_token: [output_dim]
        Returns: [1, seq_len, output_dim]
        """
        input_seq = self.pos_encoder(self.input_projection(input_seq))  # [1, T_in, d_model]
        memory = self.transformer.encoder(input_seq)

        tgt = start_token.unsqueeze(0).unsqueeze(0)  # [1, 1, output_dim]
        outputs = []

        for t in range(seq_len):
            tgt_proj = self.output_projection(tgt)        # [1, t+1, d_model]
            tgt_proj = self.pos_encoder(tgt_proj)         # positional encoding

            tgt_mask = torch.triu(torch.full((t + 1, t + 1), float('-inf'), device=input_seq.device), diagonal=1)

            decoder_out = self.transformer.decoder(
                tgt=tgt_proj,
                memory=memory,
                tgt_mask=tgt_mask
            )

            pred = self.regressor(decoder_out[:, -1:])  # predict last token
            outputs.append(pred)
            tgt = torch.cat([tgt, pred], dim=1)

        return torch.cat(outputs, dim=1)  # [1, seq_len, output_dim]
