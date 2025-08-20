import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

# --- Abstract Base Class for Preprocessing ---
class BasePreprocessing(ABC):
    def __init__(self, filename=None, n_gram_size=1, pad_token='<PAD>'):
        self.n_gram_size = n_gram_size
        self.pad_token = pad_token
        self.vocab = None
        self.idx_to_char = None
        self.char_to_idx = None
        self.vocab_size = 0

    @abstractmethod
    def process(self, *args, **kwargs):
        pass




class EmbeddingKMERPreprocessing(BasePreprocessing):
    def __init__(self,filename,n_gram_size=1,pad_token='<PAD>'):
        super().__init__(filename, n_gram_size, pad_token)

        all_tokens_in_data = set()
        self.max_kmers_size = 0
        with open(filename, 'r') as f:
            for line in f:
                kmers = self.generate_kmers(line, self.n_gram_size)
                self.max_kmers_size = max(len(kmers),self.max_kmers_size)
                all_tokens_in_data = all_tokens_in_data | set(kmers)

        self.vocab = {kmer: i+1 for i,kmer in enumerate(sorted(list(all_tokens_in_data)))}
        self.vocab[pad_token] = 0
    def generate_kmers(self,seq, k):
        """Generates non-overlapping k-mers from a sequence."""
        if len(seq) < k:
            return []  # Return empty list if sequence is shorter than k-mer size
        return [seq[i:i + k] for i in range(0, len(seq) - k + 1, k)]

    def process(self, sequence, max_len):
        numerical_seq = [self.vocab.get(char, self.vocab[self.pad_token]) for char in self.generate_kmers(sequence,self.n_gram_size)]
        padded = np.pad(numerical_seq, (max_len - len(numerical_seq), 0), 'constant',
                                    constant_values=self.vocab[self.pad_token])

        return torch.tensor(padded, dtype=torch.long)
class EmbeddingPreprocessing(BasePreprocessing):
    def __init__(self, filename, n_gram_size=1, pad_token='<PAD>'):
        super().__init__(filename, n_gram_size, pad_token)

        all_tokens_in_data = set()
        with open(filename, 'r') as f:
            for line in f:
                all_tokens_in_data.update(line.strip())
        print(f"All tokens characters found: {sorted(list(all_tokens_in_data))}")
        self.vocab = {char: i + 1 for i, char in enumerate(sorted(list(all_tokens_in_data)))}
        self.vocab[pad_token] = 0  # Pad token at index 0
        self.idx_to_char = {i: char for char, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)  # Update vocab_size
        print(f"Embedding Vocabulary size: {self.vocab_size}")

    def process(self, sequence, max_len):
        if self.n_gram_size > 1:
            processed_windows = []
            # Iterate to create all possible windows including padded ones at the start
            for i in range(len(sequence) + self.n_gram_size - 1):
                window = []
                # Extract characters for the current window
                for j in range(self.n_gram_size):
                    char_idx = i - (self.n_gram_size - 1) + j
                    if 0 <= char_idx < len(sequence):
                        window.append(self.vocab.get(sequence[char_idx], self.vocab[self.pad_token]))
                    else:
                        window.append(self.vocab[self.pad_token])  # Pad if outside sequence bounds
                processed_windows.append(window)

            # Pad the list of windows from the left to reach max_len
            pad_window = [self.vocab[self.pad_token]] * self.n_gram_size
            while len(processed_windows) < max_len:
                processed_windows.insert(0, pad_window)
            processed_windows = processed_windows[-max_len:]  # Truncate from the left if too long

            return torch.tensor(processed_windows, dtype=torch.long)

        else:  # n_gram_size is 1 (standard character embedding)
            numerical_seq = [self.vocab.get(char, self.vocab[self.pad_token]) for char in sequence]
            # Pad from the left
            padded = np.pad(numerical_seq, (max_len - len(numerical_seq), 0), 'constant',
                            constant_values=self.vocab[self.pad_token])
            return torch.tensor(padded, dtype=torch.long)


class OneHotPreprocessing(BasePreprocessing):
    def __init__(self, vocab_chars, n_gram_size=1, pad_token='<PAD>'):
        super().__init__(vocab_chars, n_gram_size, pad_token)
        # Create a vocabulary including the pad token for one-hot indexing
        all_chars = sorted(list(set(vocab_chars).union({pad_token})))
        self.char_to_idx = {char: i for i, char in enumerate(all_chars)}
        self.vocab_size = len(self.char_to_idx)
        print(f"One-Hot Vocabulary size: {self.vocab_size}")

    def process(self, sequence, max_len):
        if self.n_gram_size > 1:
            processed_windows_one_hot = []
            for i in range(len(sequence) + self.n_gram_size - 1):
                window_chars = []
                # Get characters for the current window, padding from left
                for j in range(self.n_gram_size):
                    char_idx = i - (self.n_gram_size - 1) + j
                    if 0 <= char_idx < len(sequence):
                        window_chars.append(sequence[char_idx])
                    else:
                        window_chars.append(self.pad_token)

                # Create a concatenated one-hot vector for the current window
                window_one_hot = torch.zeros(self.vocab_size * self.n_gram_size, dtype=torch.float)
                for k, char in enumerate(window_chars):
                    char_idx_in_vocab = self.char_to_idx.get(char, self.char_to_idx[self.pad_token])
                    start_idx = k * self.vocab_size
                    window_one_hot[start_idx + char_idx_in_vocab] = 1.0
                processed_windows_one_hot.append(window_one_hot)

            # Pad the list of one-hot windows from the left
            pad_one_hot_window = torch.zeros(self.vocab_size * self.n_gram_size, dtype=torch.float)
            while len(processed_windows_one_hot) < max_len:
                processed_windows_one_hot.insert(0, pad_one_hot_window)
            processed_windows_one_hot = processed_windows_one_hot[-max_len:]  # Truncate if too long

            # Stack them to get (num_windows, vocab_size * n_gram_size) and then permute for Conv1d
            # Output shape for Conv1d: (batch_size, channels, length) -> (batch_size, vocab_size*n_gram_size, max_len)
            # When batch_size=1, it will be (vocab_size*n_gram_size, max_len)
            return torch.stack(processed_windows_one_hot, dim=0).T  # Transpose to (channels, length) for a single item

        else:  # n_gram_size is 1 (standard One-Hot encoding)
            # Create a one-hot matrix for the sequence
            # (sequence_length, vocab_size)
            one_hot_sequence = torch.zeros((len(sequence), self.vocab_size), dtype=torch.float)
            for i, char in enumerate(sequence):
                idx = self.char_to_idx.get(char, self.char_to_idx[self.pad_token])
                one_hot_sequence[i, idx] = 1.0

            # Pad the one-hot sequence
            # Desired output for Conv1d: (channels, length) -> (vocab_size, max_len)
            # Pad from the left
            padded_one_hot = torch.zeros((max_len, self.vocab_size), dtype=torch.float)
            # Calculate where to start copying the sequence to achieve left padding
            start_idx = max_len - len(sequence)
            if start_idx < 0:  # If sequence is longer than max_len, truncate from left
                one_hot_sequence = one_hot_sequence[-max_len:]
                start_idx = 0
            padded_one_hot[start_idx:start_idx + len(one_hot_sequence)] = one_hot_sequence

            # Transpose to (vocab_size, max_len) for Conv1d input (C, L)
            return padded_one_hot.T


# --- Shared Embedding Preprocessing Class ---
class PreprocessingSharedEmbedding(BasePreprocessing):
    def __init__(self, rna_file, rbp_file, n_gram_size=1, step_size=1, pad_token='<PAD>'):
        super().__init__(filename=None, n_gram_size=n_gram_size, pad_token=pad_token)
        self.step_size = step_size
        self.rna_file = rna_file
        self.rbp_file = rbp_file

        tokens = set()

        with open(rna_file, 'r') as f:
            for line in f:
                line = line.strip()
                if n_gram_size > 1:
                    kmers = self.generate_kmers(line, n_gram_size, step_size)
                    tokens.update(kmers)
                else:
                    tokens.update(list(line))

        with open(rbp_file, 'r') as f:
            for line in f:
                tokens.update(list(line.strip()))

        tokens = sorted(list(tokens))
        self.vocab = {token: i + 1 for i, token in enumerate(tokens)}
        self.vocab[pad_token] = 0

        self.idx_to_token = {i: token for token, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        print(f"[SharedEmbedding] Vocab size: {self.vocab_size}")

    def generate_kmers(self, seq, k, step):
        if len(seq) < k:
            return []
        return [seq[i:i + k] for i in range(0, len(seq) - k + 1, step)]

    def encode_sequence(self, sequence, use_kmer=True):
        if use_kmer and self.n_gram_size > 1:
            tokens = self.generate_kmers(sequence, self.n_gram_size, self.step_size)
        else:
            tokens = list(sequence)
        return [self.vocab.get(tok, self.vocab[self.pad_token]) for tok in tokens]

    def process(self, rna_sequence, rbp_sequence, rna_max_len, rbp_max_len):
        rna_encoded = self.encode_sequence(rna_sequence, use_kmer=True)
        rbp_encoded = self.encode_sequence(rbp_sequence, use_kmer=False)

        rna_padded = rna_encoded + [self.vocab[self.pad_token]] * (rna_max_len - len(rna_encoded))
        rna_padded = rna_padded[:rna_max_len]

        rbp_padded = rbp_encoded + [self.vocab[self.pad_token]] * (rbp_max_len - len(rbp_encoded))
        rbp_padded = rbp_padded[:rbp_max_len]

        rna_tensor = torch.tensor(rna_padded, dtype=torch.long)
        rbp_tensor = torch.tensor(rbp_padded, dtype=torch.long)

        rna_mask = (rna_tensor != self.vocab[self.pad_token]).long()
        rbp_mask = (rbp_tensor != self.vocab[self.pad_token]).long()

        return rna_tensor, rbp_tensor, rna_mask, rbp_mask

# --- Modified CustomDataset Class ---
class CustomDataset(Dataset):
    def __init__(self, rna_file_path, amino_file_path, scores_file_path='',
                 rna_max_len=41, amino_max_len=912, expected_score_dim=200,
                 preprocessing_rna=None, preprocessing_amino=None,score_transform=None):
        self.expected_score_dim = expected_score_dim
        self.rna_file_path = rna_file_path
        self.amino_file_path = amino_file_path
        self.scores_file_path = scores_file_path
        self.rna_max_len = rna_max_len
        self.amino_max_len = amino_max_len
        self.preprocessing_rna = preprocessing_rna
        self.preprocessing_amino = preprocessing_amino
        self.score_transform = score_transform


        self._load_data()
        self._check_preprocessing_objects()

    def _check_preprocessing_objects(self):
        if not isinstance(self.preprocessing_rna, BasePreprocessing):
            raise TypeError("preprocessing_rna must be an instance of BasePreprocessing or its subclasses.")
        if not isinstance(self.preprocessing_amino, BasePreprocessing):
            raise TypeError("preprocessing_amino must be an instance of BasePreprocessing or its subclasses.")

    def _load_data(self):
        print("Loading RNA sequences...")
        with open(self.rna_file_path, 'r') as f:
            self.rna_sequences = [line.strip() for line in f]

        print("Loading Amino acid sequences...")
        with open(self.amino_file_path, 'r') as f:
            self.amino_sequences = [line.strip() for line in f]

        self.scores = []
        self.has_scores = False

        if self.scores_file_path and self.scores_file_path != '':
            print("Loading Scores...")
            try:
                self.scores = np.loadtxt(self.scores_file_path, delimiter='\t')
                self.has_scores = True
            except FileNotFoundError:
                print(f"Warning: Scores file not found at {self.scores_file_path}. Proceeding without scores.")
            except Exception as e:
                print(f"Error loading scores file: {e}. Proceeding without scores.")

        self.num_samples = len(self.rna_sequences)
        if len(self.amino_sequences) == 0:
            raise ValueError("Amino acid sequences file is empty or not found.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rna_sequence = self.rna_sequences[idx]
        amino_idx = idx % len(self.amino_sequences)
        amino_sequence = self.amino_sequences[amino_idx]

        if isinstance(self.preprocessing_rna, PreprocessingSharedEmbedding):
            rna_tensor, rbp_tensor, rna_mask, rbp_mask = self.preprocessing_rna.process(
                rna_sequence, amino_sequence, self.rna_max_len, self.amino_max_len
            )
        else:
            rna_tensor = self.preprocessing_rna.process(rna_sequence, self.rna_max_len)
            rbp_tensor = self.preprocessing_amino.process(amino_sequence, self.amino_max_len)
            pad_rna = self.preprocessing_rna.vocab[self.preprocessing_rna.pad_token]
            pad_rbp = self.preprocessing_amino.vocab[self.preprocessing_amino.pad_token]
            rna_mask = (rna_tensor != pad_rna).long()
            rbp_mask = (rbp_tensor != pad_rbp).long()

        if self.has_scores:
            raw_score = self.scores[idx][amino_idx]
            if self.score_transform:
                normalized_score = self.score_transform.transform([[raw_score]])[0][0]
            else:
                normalized_score = raw_score
            score_tensor = torch.tensor(normalized_score, dtype=torch.float)
        else:
            score_tensor = torch.zeros(self.expected_score_dim, dtype=torch.float)
        return (rna_tensor, rbp_tensor, rna_mask, rbp_mask), score_tensor
