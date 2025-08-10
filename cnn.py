import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.utils import plot_model
from keras.regularizers import l2
import keras_tuner

import data
import model

SAVED_MODEL_PATH = Path("cnn_model.keras")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
assert(len(tf.config.list_physical_devices('GPU')) > 0)
print("GPU is available and being used.")

padded_rbp_kmer_sequences, padded_validation_rbp_kmer_sequences, rbp_embedding_dim = data.load_rbp_data()
padded_rna_kmer_sequences, padded_validation_rna_kmer_sequences, rna_embedding_dim = data.load_rna_data()
binding_data, validation_binding_data = data.load_binding_data()
binding_weights, validation_binding_weights = data.calculate_weights(binding_data, validation_binding_data)


# Generate all possible (rna_idx, rbp_idx) combinations
train_indices = [(i, j) for i in range(padded_rna_kmer_sequences.shape[0]) for j in range(padded_rbp_kmer_sequences.shape[0])]
validation_indices = [(i, j) for i in range(padded_validation_rna_kmer_sequences.shape[0]) for j in range(padded_validation_rbp_kmer_sequences.shape[0])]
print(f"Total samples (combinations) generated: {len(train_indices)}")
print(f"Total validation samples (combinations) generated: {len(validation_indices)}")

if SAVED_MODEL_PATH.exists():
    print("\n--- Found a saved Model ---")
    sys.exit(1)

print("\n--- Creating Model ---")
class CNNHyperModel(keras_tuner.HyperModel):
    def my_build(self, hp):
        # use_embedding = hp.Boolean('use_embedding', default=True)
        # if use_embedding:
        #     embedding_dim = hp.Int('embedding_dim', min_value=16, max_value=256, step=16)
        use_embedding = False
        use_normalization = hp.Boolean('use_normalization', default=True)
        dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        activation = hp.Choice("activation", ["relu", "tanh"])
        number_of_filters = hp.Int('filters', min_value=64, max_value=128, step=16)
        weight_decay = hp.Float("weight_decay", min_value=1e-6, max_value=1e-2, sampling="log", default=0)
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        optimizer = hp.Choice("optimizer", ["adam", "rmsprop"])

        rna_input_shape = padded_rna_kmer_sequences.shape[1:]
        rna_input = keras.Input(shape=rna_input_shape, name='rna_input')
        if use_embedding:
            rna_input = layers.Embedding(
                input_dim=rna_embedding_dim,
                output_dim=embedding_dim,
                trainable=True,
                mask_zero=True # Ensures padded zeros are ignored
            )(rna_input)

        rna_conv1 = layers.Conv1D(number_of_filters, kernel_size=hp.Int('rna_kernel_size_1', 3, 10),
                                  padding='same', kernel_regularizer=l2(.01),
                                  input_shape=rna_input_shape)(rna_input)
        if use_normalization:
            rna_conv1 = layers.BatchNormalization()(rna_conv1)
        rna_relu1 = layers.Activation(activation)(rna_conv1)
        rna_pooling = layers.MaxPooling1D(pool_size=4)(rna_relu1)
        rna_drop1 = layers.Dropout(dropout)(rna_pooling)

        rna_conv2 = layers.Conv1D(number_of_filters, kernel_size=hp.Int('rna_kernel_size_2', min_value=3, max_value=10),
                                  padding='same', kernel_regularizer=l2(.01))(rna_drop1)
        if use_normalization:
            rna_conv2 = layers.BatchNormalization()(rna_conv2)
        rna_relu2 = layers.Activation(activation)(rna_conv2)
        rna_pooling2 = layers.MaxPooling1D(pool_size=4)(rna_relu2)
        rna_drop2 = layers.Dropout(dropout)(rna_pooling2)

        # --- RBP Branch
        rbp_input_shape = padded_rbp_kmer_sequences.shape[1:]
        rbp_input = keras.Input(shape=rbp_input_shape, name='rbp_input')
        if use_embedding:
            rbp_input = layers.Embedding(
                input_dim=rbp_embedding_dim,
                output_dim=embedding_dim,
                trainable=True,
                mask_zero=True # Ensures padded zeros are ignored
            )(rbp_input)
        rbp_conv1 = layers.Conv1D(number_of_filters, kernel_size=hp.Int('rbp_kernel_size_1', min_value=3, max_value=10),
                                  padding='same', kernel_regularizer=l2(.01),
                                  input_shape=rbp_input_shape)(rbp_input)
        if use_normalization:
            rbp_conv1 = layers.BatchNormalization()(rbp_conv1)
        rbp_relu1 = layers.Activation(activation)(rbp_conv1)
        rbp_pooling = layers.MaxPooling1D(pool_size=hp.Int('rbp_pool_size_1', min_value=2, max_value=16))(rbp_relu1)
        rbp_drop1 = layers.Dropout(dropout)(rbp_pooling)

        rbp_conv2 = layers.Conv1D(number_of_filters, kernel_size=hp.Int('rbp_kernel_size_2', min_value=3, max_value=10),
                                  padding='same', kernel_regularizer=l2(.01))(rbp_drop1)
        if use_normalization:
            rbp_conv2 = layers.BatchNormalization()(rbp_conv2)
        rbp_relu2 = layers.Activation(activation)(rbp_conv2)
        rbp_pooling2 = layers.MaxPooling1D(pool_size=hp.Int('rbp_pool_size_2', min_value=2, max_value=16))(rbp_relu2)
        rbp_drop2 = layers.Dropout(dropout)(rbp_pooling2)

        rna_flat = layers.Flatten()(rna_drop2)
        rbp_flat = layers.Flatten()(rbp_drop2)
        concatenated = layers.concatenate([rna_flat, rbp_flat], name='concatenated_vectors')

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

    def get_data_generators(self, batch_size):
        train_generator = data.DataGenerator(
            train_indices,
            padded_rna_kmer_sequences,
            padded_rbp_kmer_sequences,
            binding_data,
            binding_weights,
            batch_size,
            weighted=False,
            shuffle=True
        )

        test_generator = data.DataGenerator(
            validation_indices,
            padded_validation_rna_kmer_sequences,
            padded_validation_rbp_kmer_sequences,
            validation_binding_data,
            validation_binding_weights,
            1024,
            weighted=False,
            shuffle=False  # No need to shuffle test data
        )
        return train_generator, test_generator

    def fit(self, hp, model, **kwargs):
        batch_size = hp.Choice("batch_size", [32, 64, 128, 256, 512, 1024])
        epochs = hp.Int("epochs", min_value=3, max_value=30)
        train_generator, test_generator = self.get_data_generators(batch_size)
        history = model.fit(
            train_generator,
            validation_data=test_generator,
            shuffle=True,
            epochs=epochs,
            **kwargs
        )
        self.logging(model, test_generator)
        return history


# Initialize tuner
tuner = keras_tuner.RandomSearch(
    CNNHyperModel(),
    objective='val_loss',
    max_trials=200,
    executions_per_trial=1,
    directory='random_search',
)

test_generator = data.DataGenerator(
    validation_indices,
    padded_validation_rna_kmer_sequences,
    padded_validation_rbp_kmer_sequences,
    validation_binding_data,
    validation_binding_weights,
    1024,
    weighted=False,
    shuffle=False  # No need to shuffle test data
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