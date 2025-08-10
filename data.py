from tensorflow import keras
import numpy as np
import string
from gensim.models import FastText

RBP_KMER_SIZE = 3
RNA_KMER_SIZE = 3


def load_sequences(filepath):
    """Loads sequences from a text file, one sequence per line."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f]

def load_binding_scores(filepath):
    """Loads binding strength data from a space/tab-separated file."""
    return np.loadtxt(filepath)

def generate_kmers(sequence, k=3):
    """Generates non-overlapping k-mers from a sequence."""
    if len(sequence) < k:
        return [] # Return empty list if sequence is shorter than k-mer size
    return [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, k)]


def one_hot_encode_rna(seq):
    mapping = dict(zip("ACUG", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

def one_hot_encode_rbp(seq):
    mapping = dict(zip(string.ascii_uppercase, range(len(string.ascii_uppercase))))
    seq2 = [mapping[i] for i in seq]
    return np.eye(len(mapping))[seq2]


def load_rbp_data(embedding_dim: int):
    rbp_sequences = load_sequences('sample_data2/train_rbps2_seq.170.txt')
    validation_rbp_sequences = load_sequences('sample_data2/validation_rbps2_seq.30.txt')
    print(f"Loaded {len(rbp_sequences)} total RBP sequences.")
    print(f"Loaded {len(validation_rbp_sequences)} total RBP sequences for validation set.")

    # Generate k-mer sequences for RBP
    rbp_kmer_sequences = [generate_kmers(seq, k=RBP_KMER_SIZE) for seq in rbp_sequences]
    validation_rbp_kmer_sequences = [generate_kmers(seq, k=RBP_KMER_SIZE) for seq in validation_rbp_sequences]

    all_rbp_kmers = [kmer for seq_kmers in rbp_kmer_sequences for kmer in seq_kmers]
    rbp_kmer_vocab = sorted(list(set(all_rbp_kmers)))
    # Map RBP k-mers to integers starting from 1, reserving 0 for padding and len + 1 for unknown.
    rbp_unknown_token_index = len(rbp_kmer_vocab) + 1
    rbp_kmer_to_int = {kmer: i + 1 for i, kmer in enumerate(rbp_kmer_vocab)}

    # encoded_rbp_kmer_sequences = [one_hot_encode_rbp(seq) for seq in rbp_sequences]
    # encoded_validation_rbp_kmer_sequences = [one_hot_encode_rbp(seq) for seq in validation_rbp_sequences]
    # Encode RBP k-mer sequences into integer sequences
    encoded_rbp_kmer_sequences = [
        [rbp_kmer_to_int.get(kmer, rbp_unknown_token_index) for kmer in seq_kmers]  # Use .get(kmer, 0) for safety
        for seq_kmers in rbp_kmer_sequences
    ]
    encoded_validation_rbp_kmer_sequences = [
        [rbp_kmer_to_int.get(kmer, rbp_unknown_token_index) for kmer in seq_kmers]  # Use .get(kmer, 0) for safety
        for seq_kmers in validation_rbp_kmer_sequences
    ]

    # Determine max length for padded RBP k-mer sequences
    max_rbp_kmer_len = max(len(seq) for seq in encoded_rbp_kmer_sequences) if encoded_rbp_kmer_sequences else 0
    # Pad RBP k-mer sequences
    padded_rbp_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
        encoded_rbp_kmer_sequences, maxlen=max_rbp_kmer_len, padding='post'
    )
    padded_validation_rbp_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
        encoded_validation_rbp_kmer_sequences, maxlen=max_rbp_kmer_len, padding='post'
    )
    print(f"Max RBP k-mer sequence length: {max_rbp_kmer_len}")
    print(f"Number of unique RBP k-mers (vocabulary size): {len(rbp_kmer_vocab)}")
    print(f"Shape of padded RBP k-mer sequences: {padded_rbp_kmer_sequences.shape}")
    print(f"Shape of padded RBP k-mer sequences for validation: {padded_validation_rbp_kmer_sequences.shape}")

    print("Training FastText model for RBP k-mers...")
    fasttext_model_rbp = FastText(
        sentences=rbp_kmer_sequences,
        vector_size=embedding_dim,
        window=25,
        min_count=1, # Ensures all k-mers are considered
        sg=0, # CBOW
        min_n=2,
        max_n=3
    )
    print("FastText model training complete for RBP.")

    # Create embedding matrix for Keras RBP Embedding layer
    # Size: (vocab_size + 1 (for padding) + 1 (for unknown), embedding_dim)
    embedding_matrix_rbp = np.zeros((len(rbp_kmer_vocab) + 2, embedding_dim))
    for kmer, i in rbp_kmer_to_int.items():
        if kmer in fasttext_model_rbp.wv:
            embedding_matrix_rbp[i] = fasttext_model_rbp.wv[kmer]
    print(f"Shape of RBP embedding matrix: {embedding_matrix_rbp.shape}")

    # return padded_rbp_kmer_sequences, padded_validation_rbp_kmer_sequences, len(rbp_kmer_vocab) + 2
    return padded_rbp_kmer_sequences, padded_validation_rbp_kmer_sequences, max_rbp_kmer_len, embedding_matrix_rbp


def load_rna_data(embedding_dim: int):
    rna_sequences = load_sequences('sample_data2/train_rna_seq.102578.txt')
    validation_rna_sequences = load_sequences('sample_data2/validation_rna_seq.18100.txt')
    print(f"Loaded {len(rna_sequences)} total RNA sequences.")
    print(f"Loaded {len(validation_rna_sequences)} total RNA sequences for validation set.")

    # Generate k-mer sequences for RNA
    rna_kmer_sequences = [generate_kmers(seq, k=RNA_KMER_SIZE) for seq in rna_sequences]
    validation_rna_kmer_sequences = [generate_kmers(seq, k=RNA_KMER_SIZE) for seq in validation_rna_sequences]

    all_rna_kmers = [kmer for seq_kmers in rna_kmer_sequences for kmer in seq_kmers]
    rna_kmer_vocab = sorted(list(set(all_rna_kmers)))
    # Map k-mers to integers starting from 1, reserving 0 for padding and len + 1 for unknown.
    rna_unknown_token_index = len(rna_kmer_vocab) + 1
    rna_kmer_to_int = {kmer: i + 1 for i, kmer in enumerate(rna_kmer_vocab)}

    # encoded_rna_kmer_sequences = [one_hot_encode_rna(seq) for seq in rna_sequences]
    # encoded_validation_rna_kmer_sequences = [one_hot_encode_rna(seq) for seq in validation_rna_sequences]
    # Encode k-mer sequences into integer sequences
    encoded_rna_kmer_sequences = [
        [rna_kmer_to_int.get(kmer, rna_unknown_token_index) for kmer in seq_kmers]  # Use .get(kmer, 0) for safety
        for seq_kmers in rna_kmer_sequences
    ]
    encoded_validation_rna_kmer_sequences = [
        [rna_kmer_to_int.get(kmer, rna_unknown_token_index) for kmer in seq_kmers]  # Use .get(kmer, 0) for safety
        for seq_kmers in validation_rna_kmer_sequences
    ]

    # Determine max length for padded k-mer sequences
    max_rna_kmer_len = max(len(seq) for seq in encoded_rna_kmer_sequences)
    # Pad k-mer sequences
    padded_rna_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
        encoded_rna_kmer_sequences, maxlen=max_rna_kmer_len, padding='post'
    )

    padded_validation_rna_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
        encoded_validation_rna_kmer_sequences, maxlen=max_rna_kmer_len, padding='post'
    )

    print(f"Max RNA k-mer sequence length: {max_rna_kmer_len}")
    print(f"Number of unique RNA k-mers (vocabulary size): {len(rna_kmer_vocab)}")
    print(f"Shape of padded RNA k-mer sequences: {padded_rna_kmer_sequences.shape}")
    print(f"Shape of padded RNA k-mer sequences for validation: {padded_validation_rna_kmer_sequences.shape}")

    print("Training FastText model for RNA k-mers...")
    fasttext_model_rna = FastText(
        sentences=rna_kmer_sequences,
        vector_size=embedding_dim,
        window=25,
        min_count=1,
        sg=0,
        min_n=2,
        max_n=3
    )
    print("FastText model training complete for RNA.")

    # Create embedding matrix for Keras RNA Embedding layer
    # Size: (vocab_size + 1 (for padding) + 1 (for unknown), embedding_dim)
    embedding_matrix_rna = np.zeros((len(rna_kmer_vocab) + 2, embedding_dim))
    for kmer, i in rna_kmer_to_int.items():
        if kmer in fasttext_model_rna.wv:
            embedding_matrix_rna[i] = fasttext_model_rna.wv[kmer]
    print(f"Shape of RNA embedding matrix: {embedding_matrix_rna.shape}")

    # return padded_rna_kmer_sequences, padded_validation_rna_kmer_sequences, len(rna_kmer_vocab) + 2
    return padded_rna_kmer_sequences, padded_validation_rna_kmer_sequences, max_rna_kmer_len, embedding_matrix_rna

def load_binding_data():
    binding_data = load_binding_scores('sample_data2/train_scores.102578.txt')
    validation_binding_data = load_binding_scores('sample_data2/validation_scores.18100.txt')
    print(f"Loaded binding data matrix of shape: {binding_data.shape}")
    print(f"Loaded binding data for validation set matrix of shape: {validation_binding_data.shape}")
    return binding_data, validation_binding_data


def calculate_weights(binding_data, validation_binding_data):
    mean_data = binding_data.mean()
    mean_validation_data = validation_binding_data.mean()
    binding_weights = 1 + np.abs(binding_data - mean_data) * 2
    validation_binding_weights = 1 + np.abs(validation_binding_data - mean_validation_data) * 2
    return binding_weights, validation_binding_weights


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_indices, padded_rna_seqs, padded_rbp_seqs, binding_data, weights, batch_size, weighted=True, shuffle=True):
        self.list_indices = list_indices # List of (rna_idx, rbp_idx) tuples
        self.padded_rna_seqs = padded_rna_seqs
        self.padded_rbp_seqs = padded_rbp_seqs
        self.binding_data = binding_data
        self.weights = weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weighted = weighted
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_indices) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices of the batch within self.list_indices
        indices_in_batch = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Get the actual (rna_idx, rbp_idx) tuples for the current batch
        batch_combination_indices = [self.list_indices[k] for k in indices_in_batch]

        # Generate data for this batch
        if not self.weighted:
            X_rna, X_rbp, y = self.__data_generation(batch_combination_indices)
            return {'rna_input': X_rna, 'rbp_input': X_rbp}, y

        X_rna, X_rbp, y, sample_weights = self.__data_generation(batch_combination_indices)
        return {'rna_input': X_rna, 'rbp_input': X_rbp}, y, sample_weights

    def on_epoch_end(self):
        """Updates indices after each epoch"""
        self.indices = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_combination_indices):
        """Generates data containing batch_size samples"""
        # Pre-allocate arrays for efficiency
        X_rna_batch = np.empty((len(batch_combination_indices), ) + self.padded_rna_seqs.shape[1:], dtype=self.padded_rna_seqs.dtype)
        X_rbp_batch = np.empty((len(batch_combination_indices), ) + self.padded_rbp_seqs.shape[1:], dtype=self.padded_rbp_seqs.dtype)
        y_batch = np.empty((len(batch_combination_indices),), dtype=self.binding_data.dtype)

        for i, (rna_idx, rbp_idx) in enumerate(batch_combination_indices):
            X_rna_batch[i] = self.padded_rna_seqs[rna_idx]
            X_rbp_batch[i] = self.padded_rbp_seqs[rbp_idx]
            y_batch[i] = self.binding_data[rna_idx, rbp_idx]

        if not self.weighted:
            return X_rna_batch, X_rbp_batch, y_batch

        sample_weights = np.empty((len(batch_combination_indices),), dtype=self.binding_data.dtype)
        for i, (rna_idx, rbp_idx) in enumerate(batch_combination_indices):
            sample_weights[i] = self.weights[rna_idx, rbp_idx]

        return X_rna_batch, X_rbp_batch, y_batch, sample_weights