import numpy as np
from scipy.stats import pearsonr, spearmanr # For calculating Spearman/Pearson correlation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def pearson_correlation_metric(y_true, y_pred):
    x = y_true
    y = y_pred

    # Calculate the mean of the true and predicted values
    mx = keras.backend.mean(x)
    my = keras.backend.mean(y)

    # Calculate the variance of the true and predicted values
    vx = keras.backend.var(x)
    vy = keras.backend.var(y)

    # Calculate the covariance
    covxy = keras.backend.mean(x - mx * (y - my))

    # Calculate the Pearson correlation coefficient
    corr = covxy / keras.backend.sqrt(vx * vy + keras.backend.epsilon())  # Add epsilon for stability

    # We want to maximize the correlation, so we minimize the negative correlation
    return -keras.backend.mean(corr)


def mse_loss(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1)


def logcosh_loss(y_true, y_pred):
    error = y_pred - y_true
    return keras.backend.mean(keras.backend.log((keras.backend.exp(error) + keras.backend.exp(-error))/2), axis=-1)


def hybrid_loss(alpha):
    def loss_function(y_true, y_pred):
        logcosh = logcosh_loss(y_true, y_pred)
        pearson_corr = pearson_correlation_metric(y_true, y_pred) # The function from above
        return alpha * logcosh + (1 - alpha) * pearson_corr
    return loss_function

# Custom Callback for Correlation and Metrics Logging
class CorrelationLogger(keras.callbacks.Callback):
    def __init__(self, test_generator):
        self.test_generator = test_generator

    def logging(self, test_generator):
        all_test_predictions = []
        all_test_actuals = []
        for i in range(len(test_generator)):
            inputs, actuals = test_generator.__getitem__(i)[:2]  # Get inputs and actuals (ignore sample_weights)
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

class Attention(layers.Layer):
    """
    A custom Keras Attention layer that computes a context vector from a sequence.
    It implements a form of additive attention (Bahdanau-style).
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        # Indicate that this layer supports masking, which is important for padded sequences.
        self.supports_masking = True

    def build(self, input_shape):
        # input_shape: (batch_size, sequence_length, hidden_size)
        hidden_size = input_shape[-1]

        # Weight matrix for the input sequence (H)
        self.W = self.add_weight(name='att_weight_W', shape=(hidden_size, hidden_size),
                               initializer='glorot_uniform', trainable=True)
        # Bias vector
        self.b = self.add_weight(name='att_bias_b', shape=(hidden_size,),
                               initializer='zeros', trainable=True)
        # Context vector (u) that learns what to attend to
        self.u = self.add_weight(name='att_context_u', shape=(hidden_size, 1),
                               initializer='glorot_uniform', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, sequence_length, hidden_size)

        # Step 1: Compute alignment scores
        # uit = tanh(H * W + b)
        uit = keras.backend.tanh(keras.backend.dot(inputs, self.W) + self.b)

        # ait = u^T * uit (score before softmax)
        ait = keras.backend.dot(uit, self.u)
        # Remove the last dimension (which is 1 after dot product with self.u)
        ait = keras.backend.squeeze(ait, -1) # Result shape: (batch_size, sequence_length)

        # Step 2: Apply mask to attention scores if available
        if mask is not None:
            # Keras mask is typically 1 for valid steps and 0 for padded steps.
            # We want to set scores for padded steps to a very small number
            # so that their softmax probability becomes close to zero.
            mask = keras.backend.cast(mask, keras.backend.floatx())
            # Add a large negative number to masked positions (where mask is 0)
            ait += (mask - 1) * 1e9 # (0-1)*1e9 = -1e9, (1-1)*1e9 = 0

        # Step 3: Compute attention weights (alphas) using softmax
        alphas = keras.backend.softmax(ait) # Result shape: (batch_size, sequence_length)

        # Step 4: Compute the context vector
        # Apply attention weights to the original inputs and sum them up
        # Expand alphas dimension to match inputs for element-wise multiplication
        output = inputs * keras.backend.expand_dims(alphas, -1)
        # Sum over the sequence length dimension to get the context vector
        output = keras.backend.sum(output, axis=1) # Result shape: (batch_size, hidden_size)

        # Return both the context vector and the attention weights (alphas)
        return output, alphas # Return alphas for interpretability

    def compute_output_shape(self, input_shape):
        # The output shape is now a list of two shapes: (context_vector_shape, alphas_shape)
        hidden_size = input_shape[-1]
        sequence_length = input_shape[1]
        return [(input_shape[0], hidden_size), (input_shape[0], sequence_length)]

    def get_config(self):
        # Required for saving and loading models with custom layers
        config = super(Attention, self).get_config()
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, embedding_matrix, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = embedding_matrix.shape[0]
        self.embed_dim = embedding_matrix.shape[1]
        self.token_embeddings = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            weights=[embedding_matrix], # Pre-trained embeddings
            trainable=False, # Do not fine-tune FastText embeddings
            mask_zero=True # Ensures padded zeros are ignored
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=self.embed_dim,
        )

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
        })
        return config

# Multi-Head Self-Attention Layer
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

# Transformer Encoder Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
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
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

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

# Cross-Attention Block
class CrossAttentionBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
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
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

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