import os
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, spearmanr # For calculating Spearman/Pearson correlation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from gensim.models import FastText

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and being used.")
else:
    print("No GPU detected. TensorFlow will run on CPU.")

def load_sequences(filepath):
    """Loads sequences from a text file, one sequence per line."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f]

def load_binding_data(filepath):
    """Loads binding strength data from a space/tab-separated file."""
    return np.loadtxt(filepath)

rbp_sequences = load_sequences('sample_data2/train_rbps2_seq.170.txt')
rna_sequences = load_sequences('sample_data2/train_rna_seq.102578.txt')
# Load the binding strength matrix
binding_data = load_binding_data('sample_data2/train_scores.102578.txt')

print(f"Loaded {len(rbp_sequences)} total RBP sequences.")
print(f"Loaded {len(rna_sequences)} total RNA sequences.")
print(f"Loaded binding data matrix of shape: {binding_data.shape}")

def generate_kmers(sequence, k=3):
    """Generates non-overlapping k-mers from a sequence."""
    if len(sequence) < k:
        return [] # Return empty list if sequence is shorter than k-mer size
    return [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, k)]

RBP_KMER_SIZE = 3

# Generate k-mer sequences for RBP
rbp_kmer_sequences = [generate_kmers(seq, k=RBP_KMER_SIZE) for seq in rbp_sequences]
print(len(rbp_kmer_sequences))
RNA_KMER_SIZE = 3

# Generate k-mer sequences for RNA
rna_kmer_sequences = [generate_kmers(seq, k=RNA_KMER_SIZE) for seq in rna_sequences]
print(len(rna_kmer_sequences))

all_rna_kmers = [kmer for seq_kmers in rna_kmer_sequences for kmer in seq_kmers]
rna_kmer_vocab = sorted(list(set(all_rna_kmers)))
np.save("rna_kmer_vocab.npy", rna_kmer_vocab)
# Map k-mers to integers starting from 1, reserving 0 for padding and len + 1 for unknown.
RNA_UNKNOWN_TOKEN_INDEX = len(rna_kmer_vocab) + 1
rna_kmer_to_int = {kmer: i + 1 for i, kmer in enumerate(rna_kmer_vocab)}

# Encode k-mer sequences into integer sequences
encoded_rna_kmer_sequences = [
    [rna_kmer_to_int.get(kmer, RNA_UNKNOWN_TOKEN_INDEX) for kmer in seq_kmers] # Use .get(kmer, 0) for safety
    for seq_kmers in rna_kmer_sequences
]

all_rbp_kmers = [kmer for seq_kmers in rbp_kmer_sequences for kmer in seq_kmers]
rbp_kmer_vocab = sorted(list(set(all_rbp_kmers)))
np.save("rbp_kmer_vocab.npy", rbp_kmer_vocab)
# Map RBP k-mers to integers starting from 1, reserving 0 for padding and len + 1 for unknown.
RBP_UNKNOWN_TOKEN_INDEX = len(rbp_kmer_vocab) + 1
rbp_kmer_to_int = {kmer: i + 1 for i, kmer in enumerate(rbp_kmer_vocab)}

# Encode RBP k-mer sequences into integer sequences
encoded_rbp_kmer_sequences = [
    [rbp_kmer_to_int.get(kmer, RBP_UNKNOWN_TOKEN_INDEX) for kmer in seq_kmers] # Use .get(kmer, 0) for safety
    for seq_kmers in rbp_kmer_sequences
]

# Determine max length for padded RBP k-mer sequences
max_rbp_kmer_len = max(len(seq) for seq in encoded_rbp_kmer_sequences) if encoded_rbp_kmer_sequences else 0
# Pad RBP k-mer sequences
padded_rbp_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
    encoded_rbp_kmer_sequences, maxlen=max_rbp_kmer_len, padding='post'
)

# Determine max length for padded k-mer sequences
max_rna_kmer_len = max(len(seq) for seq in encoded_rna_kmer_sequences) if encoded_rna_kmer_sequences else 0
# Pad k-mer sequences
padded_rna_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
    encoded_rna_kmer_sequences, maxlen=max_rna_kmer_len, padding='post'
)

print(f"\nMax RBP k-mer sequence length: {max_rbp_kmer_len}")
print(f"Shape of padded RBP k-mer sequences: {padded_rbp_kmer_sequences.shape}")
print(f"Number of unique RBP k-mers (vocabulary size): {len(rbp_kmer_vocab)}")

print(f"\nMax RNA k-mer sequence length: {max_rna_kmer_len}")
print(f"Shape of padded RNA k-mer sequences: {padded_rna_kmer_sequences.shape}")
print(f"Number of unique RNA k-mers (vocabulary size): {len(rna_kmer_vocab)}")

EMBEDDING_DIM = 64 # As requested for RNA, applying to RBP as well

print("\nTraining FastText model for RBP k-mers...")
fasttext_model_rbp = FastText(
    sentences=rbp_kmer_sequences,
    vector_size=EMBEDDING_DIM,
    window=25,
    min_count=1, # Ensures all k-mers are considered
    sg=0, # CBOW
    min_n=2,
    max_n=3
)
print("FastText model training complete for RBP.")

# Create embedding matrix for Keras RBP Embedding layer
# Size: (vocab_size + 1 (for padding) + 1 (for unknown), embedding_dim)
embedding_matrix_rbp = np.zeros((len(rbp_kmer_vocab) + 2, EMBEDDING_DIM))
for kmer, i in rbp_kmer_to_int.items():
    if kmer in fasttext_model_rbp.wv:
        embedding_matrix_rbp[i] = fasttext_model_rbp.wv[kmer]
print(f"Shape of RBP embedding matrix: {embedding_matrix_rbp.shape}")
np.save("embedding_matrix_rbp.npy", embedding_matrix_rbp)

print("\nTraining FastText model for RNA k-mers...")
fasttext_model_rna = FastText(
    sentences=rna_kmer_sequences,
    vector_size=EMBEDDING_DIM,
    window=25,
    min_count=1,
    sg=0,
    min_n=2,
    max_n=3
)
print("FastText model training complete for RNA.")

# Create embedding matrix for Keras RNA Embedding layer
# Size: (vocab_size + 1 (for padding) + 1 (for unknown), embedding_dim)
embedding_matrix_rna = np.zeros((len(rna_kmer_vocab) + 2, EMBEDDING_DIM))
for kmer, i in rna_kmer_to_int.items():
    if kmer in fasttext_model_rna.wv:
        embedding_matrix_rna[i] = fasttext_model_rna.wv[kmer]
print(f"Shape of RNA embedding matrix: {embedding_matrix_rna.shape}")
np.save("embedding_matrix_rna.npy", embedding_matrix_rna)

num_rna_seqs = padded_rna_kmer_sequences.shape[0]
num_rbp_seqs = padded_rbp_kmer_sequences.shape[0]

# Generate all possible (rna_idx, rbp_idx) combinations
train_indices = []
for i in range(num_rna_seqs):
    for j in range(num_rbp_seqs):
        train_indices.append((i, j))

print(f"\nTotal samples (combinations) generated: {len(train_indices)}")

validation_rbp_sequences = load_sequences('sample_data2/validation_rbps2_seq.30.txt')
validation_rna_sequences = load_sequences('sample_data2/validation_rna_seq.18100.txt')
# Load the binding strength matrix
validation_binding_data = load_binding_data('sample_data2/validation_scores.18100.txt')

print(f"Loaded {len(validation_rbp_sequences)} total RBP sequences for validation set.")
print(f"Loaded {len(validation_rna_sequences)} total RNA sequences for validation set.")
print(f"Loaded binding data for validation set matrix of shape: {validation_binding_data.shape}")

# Generate k-mer sequences for RBP
validation_rbp_kmer_sequences = [generate_kmers(seq, k=RBP_KMER_SIZE) for seq in validation_rbp_sequences]
print(len(validation_rbp_kmer_sequences))

# Generate k-mer sequences for RNA
validation_rna_kmer_sequences = [generate_kmers(seq, k=RNA_KMER_SIZE) for seq in validation_rna_sequences]
print(len(validation_rna_kmer_sequences))

# Encode k-mer sequences into integer sequences
encoded_validation_rna_kmer_sequences = [
    [rna_kmer_to_int.get(kmer, RNA_UNKNOWN_TOKEN_INDEX) for kmer in seq_kmers] # Use .get(kmer, 0) for safety
    for seq_kmers in validation_rna_kmer_sequences
]

# Encode RBP k-mer sequences into integer sequences
encoded_validation_rbp_kmer_sequences = [
    [rbp_kmer_to_int.get(kmer, RBP_UNKNOWN_TOKEN_INDEX) for kmer in seq_kmers] # Use .get(kmer, 0) for safety
    for seq_kmers in validation_rbp_kmer_sequences
]

padded_validation_rbp_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
    encoded_validation_rbp_kmer_sequences, maxlen=max_rbp_kmer_len, padding='post'
)
padded_validation_rna_kmer_sequences = keras.preprocessing.sequence.pad_sequences(
    encoded_validation_rna_kmer_sequences, maxlen=max_rna_kmer_len, padding='post'
)

print(f"Shape of padded RBP k-mer sequences: {padded_validation_rbp_kmer_sequences.shape}")
print(f"Shape of padded RNA k-mer sequences: {padded_validation_rna_kmer_sequences.shape}")

validation_num_rna_seqs = padded_validation_rna_kmer_sequences.shape[0]
validation_num_rbp_seqs = padded_validation_rbp_kmer_sequences.shape[0]

# Generate all possible (rna_idx, rbp_idx) combinations
validation_indices = []
for i in range(validation_num_rna_seqs):
    for j in range(validation_num_rbp_seqs):
        validation_indices.append((i, j))

print(f"\nTotal validation samples (combinations) generated: {len(validation_indices)}")

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_indices, padded_rna_seqs, padded_rbp_seqs, binding_data, batch_size=32, shuffle=True):
        self.list_indices = list_indices # List of (rna_idx, rbp_idx) tuples
        self.padded_rna_seqs = padded_rna_seqs
        self.padded_rbp_seqs = padded_rbp_seqs
        self.binding_data = binding_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_indices) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch within self.list_indices
        indices_in_batch = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Get the actual (rna_idx, rbp_idx) tuples for the current batch
        batch_combination_indices = [self.list_indices[k] for k in indices_in_batch]

        # Generate data for this batch
        X_rna, X_rbp, y = self.__data_generation(batch_combination_indices)
        return {'rna_input': X_rna, 'rbp_input': X_rbp}, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.list_indices))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_combination_indices):
        'Generates data containing batch_size samples'
        # Pre-allocate arrays for efficiency
        X_rna_batch = np.empty((len(batch_combination_indices), self.padded_rna_seqs.shape[1]), dtype=self.padded_rna_seqs.dtype)
        X_rbp_batch = np.empty((len(batch_combination_indices), self.padded_rbp_seqs.shape[1]), dtype=self.padded_rbp_seqs.dtype)
        y_batch = np.empty((len(batch_combination_indices),), dtype=self.binding_data.dtype)

        for i, (rna_idx, rbp_idx) in enumerate(batch_combination_indices):
            X_rna_batch[i] = self.padded_rna_seqs[rna_idx]
            X_rbp_batch[i] = self.padded_rbp_seqs[rbp_idx]
            y_batch[i] = self.binding_data[rna_idx, rbp_idx]

        return X_rna_batch, X_rbp_batch, y_batch

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

class CorrelationLogger(keras.callbacks.Callback):
    def __init__(self, test_generator):
        super().__init__()
        self.test_generator = test_generator

    def logging(self):
        all_test_predictions = []
        all_test_actuals = []
        for i in range(len(self.test_generator)):
            inputs, actuals = self.test_generator.__getitem__(i)[:2]  # Get inputs and actuals (ignore sample_weights)
            predictions_batch = self.model.predict(inputs, verbose=0).flatten()
            all_test_predictions.extend(predictions_batch)
            all_test_actuals.extend(actuals)
        all_test_predictions = np.array(all_test_predictions)
        all_test_actuals = np.array(all_test_actuals)
        # Calculate Pearson/Spearman Correlation
        pearson_corr, _ = pearsonr(all_test_actuals, all_test_predictions)
        spearman_corr, _ = spearmanr(all_test_actuals, all_test_predictions)
        print(f'Pearson Correlation: {pearson_corr}')
        print(f'Spearman Correlation: {spearman_corr}')

    def on_epoch_end(self, epoch, logs=None):
        self.logging()

from pathlib import Path

saved_model_path = Path("cross_attention.keras")
TRANSFORMER_NUM_HEADS = 4  # Number of attention heads
TRANSFORMER_FF_DIM = 128   # Hidden layer size in the transformer feed forward network
FF_DIM = 512               # Hidden layer size in feed forward network
NUM_TRANSFORMER_BLOCKS = 3 # Number of Transformer Encoder blocks
initial_epoch = 0

if saved_model_path.exists():
    print("\n--- Loading a saved Model ---")
    model = keras.models.load_model(str(saved_model_path))
elif Path("06-0.25-attention.keras").exists():
    print("\n--- Loading a saved Model ---")
    model = keras.models.load_model("06-0.25-attention.keras")
    initial_epoch = 6
else:
    print("\n--- Creating Model ---")
    rna_input = keras.Input(shape=(max_rna_kmer_len,), name='rna_input')
    # Use FastText embeddings + positional embeddings
    rna_embedding_layer = PositionalEmbedding(
        embedding_matrix=embedding_matrix_rna,
        sequence_length=max_rna_kmer_len,
        vocab_size=len(rna_kmer_vocab) + 2, # Vocabulary size +1 (for padding) +1 (for unknown)
        embed_dim=EMBEDDING_DIM,
        name='rna_positional_embedding'
    )

    rna_embedded_sequence = rna_embedding_layer(rna_input)
    rna_mask = rna_embedding_layer.compute_mask(rna_input) # Get mask from embedding layer

    rna_encoded_sequence = rna_embedded_sequence
    for i in range(NUM_TRANSFORMER_BLOCKS):
        rna_encoded_sequence = TransformerBlock(
            embed_dim=EMBEDDING_DIM,
            num_heads=TRANSFORMER_NUM_HEADS,
            ff_dim=TRANSFORMER_FF_DIM,
            name=f'rna_transformer_block_{i+1}'
        )(rna_encoded_sequence, training=True, mask=rna_mask)

    # RBP Branch
    rbp_input = keras.Input(shape=(max_rbp_kmer_len,), name='rbp_input')
    rbp_embedding_layer = PositionalEmbedding(
        embedding_matrix=embedding_matrix_rbp,
        sequence_length=max_rbp_kmer_len,
        vocab_size=len(rbp_kmer_vocab) + 2, # Vocabulary size +1 (for padding) +1 (for unknown)
        embed_dim=EMBEDDING_DIM,
        name='rbp_positional_embedding'
    )

    rbp_embedded_sequence = rbp_embedding_layer(rbp_input)
    rbp_mask = rbp_embedding_layer.compute_mask(rbp_input) # Get mask from embedding layer

    rbp_encoded_sequence = rbp_embedded_sequence
    for i in range(NUM_TRANSFORMER_BLOCKS):
        rbp_encoded_sequence = TransformerBlock(
            embed_dim=EMBEDDING_DIM,
            num_heads=TRANSFORMER_NUM_HEADS,
            ff_dim=TRANSFORMER_FF_DIM,
            name=f'rbp_transformer_block_{i+1}'
        )(rbp_encoded_sequence, training=True, mask=rbp_mask)

    # Cross-Attention for Fusion (RNA attends to RBP)
    # RNA (query) learns from RBP (key/value)
    cross_attended_rna = CrossAttentionBlock(
        embed_dim=EMBEDDING_DIM,
        num_heads=TRANSFORMER_NUM_HEADS,
        ff_dim=TRANSFORMER_FF_DIM,
        name='rna_rbp_cross_attention'
    )(query_input=rna_encoded_sequence,
      key_value_input=rbp_encoded_sequence,
      training=True,
      query_mask=rna_mask,
      key_value_mask=rbp_mask)

    # Expand masks to match the embedding dimension for element-wise multiplication
    # This zeros out the padded elements before pooling.
    rna_mask_expanded = keras.ops.cast(keras.ops.expand_dims(rna_mask, -1), cross_attended_rna.dtype)
    rbp_mask_expanded = keras.ops.cast(keras.ops.expand_dims(rbp_mask, -1), rbp_encoded_sequence.dtype)

    masked_cross_attended_rna = cross_attended_rna * rna_mask_expanded
    masked_rbp_encoded_sequence = rbp_encoded_sequence * rbp_mask_expanded

    # Pooling
    # Pool the cross-attended RNA sequence
    rna_pooled_output = layers.GlobalAveragePooling1D(name='rna_pooled')(masked_cross_attended_rna)
    # Pool the RBP encoded sequence (from its self-attention branch)
    rbp_pooled_output = layers.GlobalAveragePooling1D(name='rbp_pooled')(masked_rbp_encoded_sequence)

    # Concatenate pooled outputs
    concatenated = layers.concatenate([rna_pooled_output, rbp_pooled_output], name='concatenated_vectors')

    # Multi-Layer Perceptron (MLP) for prediction
    mlp_output = layers.Dense(FF_DIM, activation='relu', name='mlp_dense_1')(concatenated)
    mlp_output = layers.Dropout(0.3)(mlp_output) # Dropout for regularization
    mlp_output = layers.Dense(64, activation='relu', name='mlp_dense_2')(mlp_output)
    mlp_output = layers.Dropout(0.3)(mlp_output) # Dropout for regularization

    # Output layer for regression (binding strength). Linear activation for continuous output.
    output_strength = layers.Dense(1, activation='linear', name='binding_strength_output')(mlp_output)

    # Create the Keras Model
    model = keras.Model(inputs=[rna_input, rbp_input], outputs=output_strength, name='RBP_RNA_Binding_Predictor_Transformer')
    optimizer = keras.optimizers.AdamW()
    model.compile(optimizer, loss='mse', metrics=['mae'])
    model.summary()

train_generator = DataGenerator(
    train_indices,
    padded_rna_kmer_sequences,
    padded_rbp_kmer_sequences,
    binding_data,
    batch_size=64,
    shuffle=True
)

test_generator = DataGenerator(
    validation_indices,
    padded_validation_rna_kmer_sequences,
    padded_validation_rbp_kmer_sequences,
    validation_binding_data,
    batch_size=1024,
    shuffle=False # No need to shuffle test data
)
if saved_model_path.exists():
    print("\n--- Skipping saved model training ---")
# elif Path("06-0.25-attention.keras").exists():
#     print("\n--- Skipping saved model training ---")
#     logger = CorrelationLogger(test_generator)
#     logger.set_model(model)
#     logger.logging()
else:
    print("\n--- Training Model ---")
    history = model.fit(
        train_generator, # Use the training data generator
        epochs=15, # Number of training epochs. Adjust as needed.
        validation_data=test_generator, # Use the test generator for validation
        verbose=True, # Show progress bar during training
        initial_epoch=initial_epoch, # Continue training if needed
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                              verbose=1),
            keras.callbacks.ModelCheckpoint(
                '{epoch:02d}-{val_loss:.2f}-attention.keras',
                monitor='val_loss',
                save_best_only=False,
                verbose=1,
            ),
            CorrelationLogger(test_generator),
        ],
    )
    model.save(str(saved_model_path))
