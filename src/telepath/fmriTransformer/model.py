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
    

# to mask different batches of sequence to be same size
def create_padding_masks(lengths, max_len):
    r"""args:
    lengths: in one batch, the array of the lengths of the sequences [5, 5, 4]
    max_len: the maximum length in the lengths, 5 in this case
    """
    mask = torch.zeros(len(lengths), max_len, dtype = torch.bool)
    for i, l in enumerate(lengths):
        # only mask where the sequence is padded to max_length
        mask[i, l:] = True
    return mask

### FMRI Transformer Model

class FMRITransformerModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.output_projection = nn.Linear(config.output_dim, config.d_model)
        self.regressor = nn.Linear(config.d_model, config.output_dim)
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)

        self.transformer = nn.Transformer(
            d_model = config.d_model,
            nhead = config.nhead,
            num_encoder_layers = config.nlayers,
            num_decoder_layers = config.nlayers,
            dim_feedforward = config.nhid,
            dropout = config.dropout,
            batch_first = True           
        )
    
    def forward(self, input_seq, fmri_seq, src_lengths, tgt_lengths):
        r"""
        This function run the whole transformer stack on a whole batch of sequences

        Note that the input_seq, fmri_seq is padded before enter this function to make sure
        different lengths of vectors are padded to make sure in each batch, the sequences have same length
        parameters:
        input_seq: padded stimulus sequence in batch, shape: [batch, seq_len, feature_dimension]
        fmri_seq: padded fmri sequence in batch, shape: [batch, seq_len, fmri_dimension]
        src_lengths: the actual length of sequences in this input batch
        tgt_lengths: the actual length of sequences in this ouput batch
        """

        # project both input and ouput sequences to space (seq_len X model_d)
        input_seq = self.pos_encoder(self.input_projection(input_seq))
        fmri_seq = self.pos_encoder(self.output_projection(fmri_seq))

        # create masks for padded input and output
        input_mask = create_padding_masks(src_lengths, input_seq.size(1))
        output_mask = create_padding_masks(tgt_lengths, fmri_seq.size(1))

        # create casual mask to make sure in decoder self-attention no looking ahead of the sequence
        causal_mask = torch.triu(
            torch.full((fmri_seq.size(1), fmri_seq.size(1)), float('-inf')), diagonal=1
        ).to(input_seq.device)

        out = self.transformer(
            src=input_seq,
            tgt=fmri_seq,
            tgt_mask=causal_mask,
            src_key_padding_mask=input_mask,   #encoder self-attetion
            tgt_key_padding_mask=output_mask,  #decoder self-attetion
            memory_key_padding_mask=input_mask # cross-attention
        )
        return self.regressor(out)
    
    @torch.no_grad()
    def autoregressive_inference(self, input_seq, seq_len, start_token):
        r""" This helps to autoregressively inference the fmri sequence step by step
        
        But note that this only process one sequence at a time, instead of a batch of sequences

        input_seq: a sequence of input vectors shape: [1, input_seq_len, input_dim]
        seq_len: the length of inferenced fmri length, same as input_seq_len here
        start_token: should be the average fmri vector: [fmri_dim]

        Return: a predicted fmri sequence: [1, seq_len, fmro_dim]
        """

        input_seq = self.pos_encoder(self.input_projection(input_seq))
        memory = self.transformer.encoder(input_seq)

        tgt = start_token.unsqueeze(0).unsqueeze(0) # [fmri_dim] -> [1,1,fmri_dim]
        
        outputs = []
        for t in range(seq_len):
            tgt_proj = self.pos_encoder(self.output_projection(tgt))
            tgt_mask = torch.triu(
                torch.full((t+1, t+1), float('-inf'), device=input_seq.device), diagonal=1
            )
            decoder_out = self.transformer.decoder(
                tgt=tgt_proj,
                memory=memory,
                tgt_mask=tgt_mask
            )

            pred = self.regressor(decoder_out[:, -1:])
            outputs.append(pred) # the start token would not join the ouputs

            tgt = torch.cat([tgt, pred], dim=1)

        return torch.cat(outputs, dim=1)

        



# not done yet
def pad_batch(sequences):
    ...

# not done yet
def train_one_epoch(model):
    ...