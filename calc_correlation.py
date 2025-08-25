import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from tqdm import tqdm
import argparse
import datetime
import os
import pathlib

RBP_KMER_SIZE = 3
RNA_KMER_SIZE = 3
MAX_RBP_KMER_LEN = 304
MAX_RNA_KMER_LEN = 13

def load_sequences(filepath):
    """Loads sequences from a text file, one sequence per line."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f]

def generate_kmers(sequence, k=3):
    """Generates non-overlapping k-mers from a sequence."""
    if len(sequence) < k:
        return [] # Return empty list if sequence is shorter than k-mer size
    return [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, k)]

def run_gpu_check():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available and being used.")
    else:
        print("No GPU detected. TensorFlow will run on CPU.")

def generate_all_possible_pairs(padded_rbp_kmer_sequences, padded_rna_kmer_sequences):
    validation_indices = []
    for i in range(padded_rna_kmer_sequences.shape[0]):
        for j in range(padded_rbp_kmer_sequences.shape[0]):
            validation_indices.append((i, j))
    return validation_indices

def load_data(rbp_file_path: str, rna_file_path: str):
    rbp_sequences = load_sequences(rbp_file_path)
    rna_sequences = load_sequences(rna_file_path)
    print(f"Loaded {len(rbp_sequences)} total RBP sequences for prediction.")
    print(f"Loaded {len(rna_sequences)} total RNA sequences for prediction.")

    # Generate k-mer sequences for RBP and RNA
    rbp_kmer_sequences = [generate_kmers(seq, k=RBP_KMER_SIZE) for seq in rbp_sequences]
    rna_kmer_sequences = [generate_kmers(seq, k=RNA_KMER_SIZE) for seq in rna_sequences]

    # Encode k-mer sequences into integer sequences
    rna_kmer_vocab = np.load("rna_kmer_vocab.npy")
    rna_kmer_to_int = {kmer: i + 1 for i, kmer in enumerate(rna_kmer_vocab)}
    encoded_rna_kmer_sequences = [
        [rna_kmer_to_int.get(kmer, len(rna_kmer_vocab) + 1) for kmer in seq_kmers]
        for seq_kmers in rna_kmer_sequences
    ]
    padded_rna_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
        encoded_rna_kmer_sequences, maxlen=MAX_RNA_KMER_LEN, padding='post'
    )
    rbp_kmer_vocab = np.load("rbp_kmer_vocab.npy")
    rbp_kmer_to_int = {kmer: i + 1 for i, kmer in enumerate(rbp_kmer_vocab)}
    encoded_rbp_kmer_sequences = [
        [rbp_kmer_to_int.get(kmer, len(rbp_kmer_vocab) + 1) for kmer in seq_kmers]
        for seq_kmers in rbp_kmer_sequences
    ]
    padded_rbp_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
        encoded_rbp_kmer_sequences, maxlen=MAX_RBP_KMER_LEN, padding='post'
    )
    print(f"Shape of padded RBP k-mer sequences: {padded_rbp_kmer_sequences.shape}")
    print(f"Shape of padded RNA k-mer sequences: {padded_rna_kmer_sequences.shape}")
    return padded_rbp_kmer_sequences, padded_rna_kmer_sequences


class TestDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_indices, padded_rna_seqs, padded_rbp_seqs, batch_size):
        self.list_indices = list_indices # List of (rna_idx, rbp_idx) tuples
        self.padded_rna_seqs = padded_rna_seqs
        self.padded_rbp_seqs = padded_rbp_seqs
        self.batch_size = batch_size
        self.indices = np.arange(len(self.list_indices))

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return len(self.list_indices) // self.batch_size

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indices of the batch within self.list_indices
        indices_in_batch = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Get the actual (rna_idx, rbp_idx) tuples for the current batch
        batch_combination_indices = [self.list_indices[k] for k in indices_in_batch]

        # Generate data for this batch
        X_rna, X_rbp = self.__data_generation(batch_combination_indices)
        return {'rna_input': X_rna, 'rbp_input': X_rbp}

    def __data_generation(self, batch_combination_indices):
        """ Generates data containing batch_size samples """
        # Pre-allocate arrays for efficiency
        X_rna_batch = np.empty((self.batch_size, self.padded_rna_seqs.shape[1]), dtype=self.padded_rna_seqs.dtype)
        X_rbp_batch = np.empty((self.batch_size, self.padded_rbp_seqs.shape[1]), dtype=self.padded_rbp_seqs.dtype)

        for i, (rna_idx, rbp_idx) in enumerate(batch_combination_indices):
            X_rna_batch[i] = self.padded_rna_seqs[rna_idx]
            X_rbp_batch[i] = self.padded_rbp_seqs[rbp_idx]

        return X_rna_batch, X_rbp_batch

@keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, embedding_matrix, sequence_length, vocab_size, embed_dim, position_embedding_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            trainable=False, # Do not fine-tune pre-trained embeddings
            mask_zero=True # Ensures padded zeros are ignored
        )
        if position_embedding_weights is not None:
            position_embedding_weights = [position_embedding_weights]
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim,
            weights=position_embedding_weights,
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # Use keras.ops.not_equal for compatibility with KerasTensors
        return keras.ops.not_equal(inputs, 0) # Mask padding token (0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "embedding_matrix": self.token_embeddings.get_weights()[0].tolist(),
            "position_embedding_weights": self.position_embeddings.get_weights()[0].tolist(),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["embedding_matrix"] = np.array(config["embedding_matrix"])
        config["position_embedding_weights"] = np.array(config["position_embedding_weights"])
        return super().from_config(config)

@keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.proj_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value, mask=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        if mask is not None:
            # Expand mask to match scaled_score dimensions
            # mask shape: (batch_size, 1, 1, sequence_length) for self-attention
            # or (batch_size, 1, query_seq_len, key_seq_len) for cross-attention
            # Cast mask to float32 before multiplication
            mask = tf.cast(mask, tf.float32)
            scaled_score += (mask * -1e9) # Add large negative number to masked positions

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        # inputs can be (query, key, value) for cross-attention
        # or (x) for self-attention (where query=key=value=x)
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            query_input, key_input, value_input = inputs
            # Mask for cross-attention: only key_input mask is relevant for attention calculation
            if mask is not None and isinstance(mask, (list, tuple)) and len(mask) == 3:
                key_mask = mask[1]
            else:
                key_mask = None
        else:
            query_input = key_input = value_input = inputs
            if mask is not None:
                key_mask = mask # For self-attention, the input mask is the key mask
            else:
                key_mask = None

        batch_size = tf.shape(query_input)[0]

        query = self.query_dense(query_input)
        key = self.key_dense(key_input)
        value = self.value_dense(value_input)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # Prepare mask for attention function
        attention_mask = None
        if key_mask is not None:
            # key_mask shape: (batch_size, key_sequence_length)
            # Reshape to (batch_size, 1, 1, key_sequence_length) for broadcasting
            attention_mask = tf.expand_dims(tf.expand_dims(key_mask, 1), 1)

        attention, weights = self.attention(query, key, value, attention_mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

    def compute_mask(self, inputs, mask=None):
        # For multi-head attention, the output mask should be the same as the query input mask
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            return mask[0] if mask is not None else None
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

@keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        # Self-attention part
        attn_output = self.att(inputs, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward part
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.dropout1.rate, # Assuming both dropouts have same rate
        })
        return config

@keras.utils.register_keras_serializable()
class CrossAttentionBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.cross_att = MultiHeadSelfAttention(embed_dim, num_heads) # Reusing MHA for cross-attention
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, query_input, key_value_input, training, query_mask=None, key_value_mask=None):
        # Cross-attention part: query_input attends to key_value_input
        attn_output = self.cross_att([query_input, key_value_input, key_value_input],
                                     mask=[query_mask, key_value_mask, key_value_mask])
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query_input + attn_output)

        # Feed-forward part
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_mask(self, inputs, mask=None):
        # The output mask should be the same as the query input mask
        return mask[0] if mask is not None else None

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.dropout1.rate,
        })
        return config


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference and export rbpXX.txt grouped by amino_idx."
    )
    parser.add_argument("out_dir", type=pathlib.Path, help="Output directory for CSV and rbpXX.txt files")
    parser.add_argument("rbp_file", type=str, help="Path to RBP/amino sequences file")
    parser.add_argument("rna_file", type=str, help="Path to RNA sequences file")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    # Make sure output directory exists
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    run_gpu_check()

    print("--- Loading input data ---")
    padded_rbp_kmer_sequences, padded_rna_kmer_sequences = load_data(arguments.rbp_file, arguments.rna_file)
    pair_indices = generate_all_possible_pairs(padded_rbp_kmer_sequences, padded_rna_kmer_sequences)
    print(f"Total validation samples (combinations) generated: {len(pair_indices)}")

    print("--- Loading a saved Model ---")
    model = keras.models.load_model("06-0.25-attention.keras")

    test_generator = TestDataGenerator(
        pair_indices,
        padded_rna_kmer_sequences,
        padded_rbp_kmer_sequences,
        batch_size=1024,
    )
    predictions = np.zeros((padded_rna_kmer_sequences.shape[0], padded_rbp_kmer_sequences.shape[0]), dtype=np.float64)
    for i in tqdm(range(len(test_generator) + 1)):
        indices_in_batch = test_generator.indices[i*test_generator.batch_size:(i+1)*test_generator.batch_size]
        batch_combination_indices = np.array([test_generator.list_indices[k] for k in indices_in_batch])

        predictions_batch = model.predict(test_generator[i], verbose=0).flatten()
        predictions[batch_combination_indices[:, 0], batch_combination_indices[:, 1]] = predictions_batch[:len(indices_in_batch)]

    for i in tqdm(range(padded_rbp_kmer_sequences.shape[0])):
        rna_predications = predictions[:, i]
        rbp_output_filename = f"RBP2{int(i + 1):02d}.txt"
        np.savetxt(os.path.join(arguments.out_dir, rbp_output_filename), rna_predications, fmt="%.6f")  # one score per line
        print(f"Wrote #{len(predictions)} to {rbp_output_filename}")


if __name__ == "__main__":
    start = datetime.datetime.now()
    main()
    print(f'total time: {(datetime.datetime.now() - start)}')
