import os

import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BasePreprocessing(ABC):
    """ Abstract Base Class for Preprocessing --- """
    _registry: dict = {}  # auto-register subclasses

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

    def save(self, path: str):
        """ Save the preprocessed data to disk """
        payload = {
            "__type__": "preprocessor",
            "__class__": self.__class__.__name__,
            "__version__": 1,
            "state": self._state_dict(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)

    def __init_subclass__(cls, **kwargs):
        """ Automatically register all subclasses """
        super().__init_subclass__(**kwargs)
        BasePreprocessing._registry[cls.__name__] = cls

    def _state_dict(self) -> dict:
        """ Represent this object as a dictionary """
        base = {
            "n_gram_size": self.n_gram_size,
            "pad_token": self.pad_token,
            "vocab": getattr(self, "vocab", None),
            "vocab_size": getattr(self, "vocab_size", 0),
        }
        base.update(self._extra_state())
        return base

    def _extra_state(self) -> dict:
        """ To be overridden by subclasses """
        return {}

    @classmethod
    def _from_state(cls, state: dict):
        """ Create preprocessor from state """
        obj = object.__new__(cls)
        obj.n_gram_size = state["n_gram_size"]
        obj.pad_token = state["pad_token"]
        obj.vocab = state.get("vocab")
        obj.vocab_size = state.get("vocab_size", len(obj.vocab) if obj.vocab is not None else 0)
        obj.idx_to_char = None
        obj.char_to_idx = None
        cls._load_extra_state(obj, state)
        return obj

    @classmethod
    def _load_extra_state(cls, obj, state: dict):
        """ To be overridden by subclasses """
        pass

    @staticmethod
    def load_preprocessor(path: str) -> 'BasePreprocessing':
        """ Load preprocessor from path """
        payload = torch.load(path, map_location="cpu")
        if payload.get("__type__") != "preprocessor":
            raise ValueError(f"File at {path} is not a preprocessor payload")
        class_name = payload["__class__"]
        cls = BasePreprocessing._registry.get(class_name)
        if cls is None:
            raise ValueError(f"Unknown preprocessor class: {class_name}. "
                             f"Known: {list(BasePreprocessing._registry)}")
        return cls._from_state(payload["state"])


class EmbeddingKMERPreprocessing(BasePreprocessing):
    def __init__(self, filename, n_gram_size=1, pad_token='<PAD>'):
        super().__init__(filename, n_gram_size, pad_token)

        all_tokens_in_data = set()
        self.max_kmers_size = 0
        with open(filename, 'r') as f:
            for line in f:
                kmers = self.generate_kmers(line, self.n_gram_size)
                self.max_kmers_size = max(len(kmers), self.max_kmers_size)
                all_tokens_in_data = all_tokens_in_data | set(kmers)

        self.vocab = {kmer: i + 1 for i, kmer in enumerate(sorted(list(all_tokens_in_data)))}
        self.vocab[pad_token] = 0

    def _extra_state(self):
        return {"max_kmers_size": getattr(self, "max_kmers_size", 0)}

    @classmethod
    def _load_extra_state(cls, obj, state: dict):
        obj.max_kmers_size = state.get("max_kmers_size", 0)

    def generate_kmers(self, seq, k):
        """Generates non-overlapping k-mers from a sequence."""
        if len(seq) < k:
            return []  # Return empty list if sequence is shorter than k-mer size
        return [seq[i:i + k] for i in range(0, len(seq) - k + 1, k)]

    def process(self, sequence, max_len):
        numerical_seq = [self.vocab.get(char, self.vocab[self.pad_token]) for char in
                         self.generate_kmers(sequence, self.n_gram_size)]
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

    def _extra_state(self):
        return {"idx_to_char": getattr(self, "idx_to_char", None)}

    @classmethod
    def _load_extra_state(cls, obj, state: dict):
        obj.idx_to_char = state.get("idx_to_char")
        obj.char_to_idx = None

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


class CustomDataset(Dataset):
    def __init__(
            self,
            rna_file_path,
            amino_file_path,
            scores_file_path='',
            rna_max_len=41,
            amino_max_len=912,
            expected_score_dim=200,
            preprocessing_rna=None,
            preprocessing_amino=None,
            score_transform=None
    ):
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
        self.num_samples *= len(self.amino_sequences)

    def __len__(self):
        return self.num_samples

    def get_seqs_by_index(self, idx):
        K = len(self.amino_sequences)
        rna_idx = idx // K
        rbp_idx = idx % K

        rna_sequence = self.rna_sequences[rna_idx]
        amino_sequence = self.amino_sequences[rbp_idx]
        return rna_idx, rna_sequence, rbp_idx, amino_sequence

    def __getitem__(self, idx):
        rna_idx, rna_sequence, rbp_idx, amino_sequence = self.get_seqs_by_index(idx)

        rna_tensor = self.preprocessing_rna.process(rna_sequence, self.rna_max_len)
        rbp_tensor = self.preprocessing_amino.process(amino_sequence, self.amino_max_len)
        pad_rna = self.preprocessing_rna.vocab[self.preprocessing_rna.pad_token]
        pad_rbp = self.preprocessing_amino.vocab[self.preprocessing_amino.pad_token]
        rna_mask = (rna_tensor != pad_rna).long()
        rbp_mask = (rbp_tensor != pad_rbp).long()

        if self.has_scores:
            raw_score = self.scores[rna_idx][rbp_idx]
            if self.score_transform:
                normalized_score = self.score_transform.transform([[raw_score]])[0][0]
            else:
                normalized_score = raw_score
            score_tensor = torch.tensor(normalized_score, dtype=torch.float)
        else:
            score_tensor = torch.zeros(self.expected_score_dim, dtype=torch.float)
        return (rna_tensor, rbp_tensor, rna_mask, rbp_mask), score_tensor
