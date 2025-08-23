import os

# Suppress oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.utils import plot_model
import keras_tuner

import data
import model

SAVED_MODEL_PATH = Path("attention_model.keras")
EMBEDDING_DIM = 64

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
assert(len(tf.config.list_physical_devices('GPU')) > 0)
print("GPU is available and being used.")

binding_data, validation_binding_data = data.load_binding_data()
binding_weights, validation_binding_weights = None, None

if SAVED_MODEL_PATH.exists():
    print("\n--- Found a saved Model ---")
    sys.exit(1)

# Generate all possible (rna_idx, rbp_idx) combinations
padded_rbp_kmer_sequences, padded_validation_rbp_kmer_sequences, max_rbp_kmer_len, embedding_matrix_rbp = data.load_rbp_data(EMBEDDING_DIM)
padded_rna_kmer_sequences, padded_validation_rna_kmer_sequences, max_rna_kmer_len, embedding_matrix_rna = data.load_rna_data(EMBEDDING_DIM)
train_indices = [(i, j) for i in range(padded_rna_kmer_sequences.shape[0]) for j in
                 range(padded_rbp_kmer_sequences.shape[0])]
validation_indices = [(i, j) for i in range(padded_validation_rna_kmer_sequences.shape[0]) for j in
                      range(padded_validation_rbp_kmer_sequences.shape[0])]
print(f"Total samples (combinations) generated: {len(train_indices)}")
print(f"Total validation samples (combinations) generated: {len(validation_indices)}")

test_generator = data.DataGenerator(
    validation_indices[:1024],
    padded_validation_rna_kmer_sequences,
    padded_validation_rbp_kmer_sequences,
    validation_binding_data,
    validation_binding_weights,
    512,
    weighted=False,
    shuffle=False  # No need to shuffle test data
)

class CNNHyperModel(keras_tuner.HyperModel):
    def my_build(self, hp):
        transformer_ff_dim = hp.Int("transformer_ff_dim", min_value=32, max_value=128, step=32)
        transformer_num_heads = hp.Choice("transformer_num_heads", [8, 16])
        dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        activation = hp.Choice("activation", ["relu", "tanh"])
        weight_decay = hp.Float("weight_decay", min_value=1e-6, max_value=1e-2, sampling="log", default=0)
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        optimizer = hp.Choice("optimizer", ["adam", "rmsprop"])

        rna_input = keras.Input(shape=(max_rna_kmer_len,), name='rna_input')
        # Use FastText embeddings + positional embeddings
        rna_embedding_layer = model.PositionalEmbedding(
            embedding_matrix=embedding_matrix_rna,
            sequence_length=max_rna_kmer_len,
            name='rna_positional_embedding'
        )

        rna_embedded_sequence = rna_embedding_layer(rna_input)
        rna_mask = rna_embedding_layer.compute_mask(rna_input)  # Get mask from embedding layer

        rna_encoded_sequence = rna_embedded_sequence
        for i in range(hp.Int('num_transformer_blocks', min_value=1, max_value=2)):
            rna_encoded_sequence = model.TransformerBlock(
                embed_dim=EMBEDDING_DIM,
                num_heads=transformer_num_heads,
                ff_dim=transformer_ff_dim,
                dropout=dropout,
                name=f'rna_transformer_block_{i + 1}'
            )(rna_encoded_sequence, training=True, mask=rna_mask)

        # RBP Branch
        rbp_input = keras.Input(shape=(max_rbp_kmer_len,), name='rbp_input')
        rbp_embedding_layer = model.PositionalEmbedding(
            embedding_matrix=embedding_matrix_rbp,
            sequence_length=max_rbp_kmer_len,
            name='rbp_positional_embedding'
        )

        rbp_embedded_sequence = rbp_embedding_layer(rbp_input)
        rbp_mask = rbp_embedding_layer.compute_mask(rbp_input)  # Get mask from embedding layer

        rbp_encoded_sequence = rbp_embedded_sequence
        for i in range(hp.Int('num_transformer_blocks', min_value=1, max_value=2)):
            rbp_encoded_sequence = model.TransformerBlock(
                embed_dim=EMBEDDING_DIM,
                num_heads=transformer_num_heads,
                ff_dim=transformer_ff_dim,
                dropout=dropout,
                name=f'rbp_transformer_block_{i + 1}'
            )(rbp_encoded_sequence, training=True, mask=rbp_mask)

        # Cross-Attention for Fusion (RNA attends to RBP)
        # RNA (query) learns from RBP (key/value)
        cross_attended_rna = model.CrossAttentionBlock(
            embed_dim=EMBEDDING_DIM,
            num_heads=transformer_num_heads,
            ff_dim=transformer_ff_dim,
            dropout=dropout,
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

        # Pool the cross-attended RNA sequence
        rna_pooled_output = layers.GlobalAveragePooling1D(name='rna_pooled')(masked_cross_attended_rna)
        # Pool the RBP encoded sequence (from its self-attention branch)
        rbp_pooled_output = layers.GlobalAveragePooling1D(name='rbp_pooled')(masked_rbp_encoded_sequence)

        # Concatenate pooled outputs
        concatenated = layers.concatenate([rna_pooled_output, rbp_pooled_output], name='concatenated_vectors')

        # Multi-Layer Perceptron (MLP)
        mlp_output = concatenated
        for i in range(hp.Int('num_dense_layers', min_value=1, max_value=3)):
            mlp_output = layers.Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=1024, step=32),
                activation=activation,
            )(mlp_output)
            mlp_output = layers.Dropout(dropout)(mlp_output)  # Dropout for regularization
        output_strength = layers.Dense(1, activation=activation, name='binding_strength_output')(mlp_output)

        # Create the Keras Model
        cnn_model = keras.Model(inputs=[rna_input, rbp_input], outputs=output_strength,
                                name='RBP_RNA_Binding_Predictor')
        if optimizer == "adam":
            optimizer = keras.optimizers.Adam(learning_rate, decay=weight_decay)
        else:
            optimizer = keras.optimizers.RMSprop(learning_rate, decay=weight_decay)
        logcosh = keras.losses.LogCosh()
        cnn_model.compile(optimizer=optimizer, loss=logcosh, metrics=['mae'])
        return cnn_model

    def build(self, hp):
        return self.my_build(hp)

    def fit(self, hp, model, **kwargs):
        epochs = hp.Int("epochs", min_value=3, max_value=30)
        batch_size = hp.Choice("batch_size", [32, 64, 128])

        train_generator = data.DataGenerator(
            train_indices[:1024],
            padded_rna_kmer_sequences,
            padded_rbp_kmer_sequences,
            binding_data,
            binding_weights,
            batch_size,
            weighted=False,
            shuffle=True
        )
        history = model.fit(
            train_generator,
            validation_data=test_generator,
            shuffle=True,
            epochs=epochs,
            **kwargs
        )
        return history


# Initialize tuner
tuner = keras_tuner.RandomSearch(
    CNNHyperModel(),
    objective='val_loss',
    max_trials=2,
    executions_per_trial=1,
    directory='random_search',
)
tuner.search(
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                          verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1),
        model.CorrelationLogger(test_generator),
    ],
)

# Get best model after tuning
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hp.values}")
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
best_model.save(str(SAVED_MODEL_PATH))

print("\n--- Diagnosing Prediction Bias ---")