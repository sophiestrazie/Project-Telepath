import os
import string
import numpy as np
import pandas as pd
import torch
import h5py
from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer, BertModel


from .config import LanguageConfig

class LanguageProcessor:
    def __init__(self, config: LanguageConfig):
        self.config = config
        self.device = config.device
        self.model, self.tokenizer = self.get_language_model()
    
    def get_language_model(self):
        """
        Load a pre-trained bert-base-uncased language model and its corresponding
        tokenizer.

        Parameters
        ----------
        device : torch.device
            Device on which the model will run (e.g., 'cpu' or 'cuda').

        Returns
        -------
        model : object
            Pre-trained language model.
        tokenizer : object
            Tokenizer corresponding to the language model.

        """

        ### Load the model ###
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval().to(self.device)

        ### Load the tokenizer ###
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
            do_lower_case=True)

        ### Output ###
        return model, tokenizer

    def process(self, episode_path):
        """
        Extract language features from a movie using a pre-trained language model.

        Parameters
        ----------
        episode_path : str
            Path to the movie transcripts for which the language features are
            extracted.
        model : object
            Pre-trained language model.
        tokenizer : object
            Tokenizer corresponding to the language model.
        num_used_tokens : int
            Total number of tokens that are fed to the language model for each
            chunk, including the tokens from the chunk of interest plus N tokens
            from previous chunks (the maximum allowed by the model is 510).
        kept_tokens_last_hidden_state : int
            Number of features retained for the last_hidden_state, where each
            feature corresponds to a token, starting from the most recent token.
        device : str
            Device to perform computations ('cpu' or 'gpu').
        save_dir_features : str
            Directory where the extracted language features are saved.

        Returns
        -------
        pooler_output : list
            List containing the pooler_output features for each chunk.
        last_hidden_state : list
            List containing the last_hidden_state features for each chunk

        """

        ### Load the transcript ###
        df = pd.read_csv(episode_path, sep='\t')
        df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

        ### Initialize the tokens and features lists ###
        tokens, np_tokens, pooler_output, last_hidden_state = [], [], [], []

        ### Loop over text chunks ###
        for i in tqdm(range(df.shape[0]), desc="Extracting language features"):

            ### Tokenize raw text ###
            if not df.iloc[i]["is_na"]: # Only tokenize if words were spoken during a chunk (i.e., if the chunk is not empty)
                # Tokenize raw text with puntuation (for pooler_output features)
                tr_text = df.iloc[i]["text_per_tr"]
                tokens.extend(self.tokenizer.tokenize(tr_text))
                # Tokenize without punctuation (for last_hidden_state features)
                tr_np_tokens = self.tokenizer.tokenize(
                    tr_text.translate(str.maketrans('', '', string.punctuation)))
                np_tokens.extend(tr_np_tokens)

            ### Extract the pooler_output features ###
            if len(tokens) > 0: # Only extract features if there are tokens available
                # Select the number of tokens used from the current and past chunks,
                # and convert them into IDs
                used_tokens = self.tokenizer.convert_tokens_to_ids(
                    tokens[-(self.config.num_used_tokens):])
                # IDs 101 and 102 are special tokens that indicate the beginning and
                # end of an input sequence, respectively.
                input_ids = [101] + used_tokens + [102]
                tensor_tokens = torch.tensor(input_ids).unsqueeze(0).to(self.device)
                # Extract and store the pooler_output features
                with torch.no_grad():
                    outputs = self.model(tensor_tokens)
                    pooler_output.append(outputs['pooler_output'][0].cpu().numpy())
            else: # Store NaN values if no tokes are available
                pooler_output.append(np.full(768, np.nan, dtype='float32'))

            ### Extract the last_hidden_state features ###
            if len(np_tokens) > 0: # Only extract features if there are tokens available
                np_feat = np.full((self.config.kept_tokens_last_hidden_state, 768), np.nan, dtype='float32')
                # Select the number of tokens used from the current and past chunks,
                # and convert them into IDs
                used_tokens = self.tokenizer.convert_tokens_to_ids(
                    np_tokens[-(self.config.num_used_tokens):])
                # IDs 101 and 102 are special tokens that indicate the beginning and
                # end of an input sequence, respectively.
                np_input_ids = [101] + used_tokens + [102]
                np_tensor_tokens = torch.tensor(np_input_ids).unsqueeze(0).to(self.device)
                # Extract and store the last_hidden_state features
                with torch.no_grad():
                    np_outputs = self.model(np_tensor_tokens)
                    np_outputs = np_outputs['last_hidden_state'][0][1:-1].cpu().numpy()
                tk_idx = min(self.config.kept_tokens_last_hidden_state, len(np_tokens))
                np_feat[-tk_idx:, :] = np_outputs[-tk_idx:]
                last_hidden_state.append(np_feat)
            else: # Store NaN values if no tokens are available
                last_hidden_state.append(np.full(
                    (self.config.kept_tokens_last_hidden_state, 768), np.nan, dtype='float32'))

        ### Convert the language features to float32 ###
        pooler_output = np.array(pooler_output, dtype='float32')
        last_hidden_state = np.array(last_hidden_state, dtype='float32')

        

        ### Output ###
        return pooler_output, last_hidden_state
   
    def save_features(self, pooler_output, last_hidden_state, episode_id):
        ### Save the language features ###
        out_file_language = os.path.join(
           self.config.save_dir_features, f"{episode_id}_language_features.npy")
        with h5py.File(out_file_language, 'a' if Path(out_file_language).exists() else 'w') as f:
           group = f.create_group(episode_id)
           group.create_dataset('language_pooler_output', data=pooler_output,
               dtype=np.float32)
           group.create_dataset('language_last_hidden_state',
               data=last_hidden_state, dtype=np.float32)
        print(f"Language features saved to {out_file_language}")