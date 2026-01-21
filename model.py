

from __future__ import annotations

from tensorflow import keras


def build_1d_cnn_feature_extractor(input_shape, num_classes, learning_rate: float = 1e-3) -> keras.Model:
    """1D CNN for feature extraction and classification."""
    # Hyperparameters (kept inside this function only)
    conv_filters = (32, 64)
    kernel_sizes = (5, 3)
    dense_units = 64
    dropout = 0.3

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(conv_filters[0], kernel_size=kernel_sizes[0], activation="relu")(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Conv1D(conv_filters[1], kernel_size=kernel_sizes[1], activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    net = keras.Model(inputs, outputs, name="1D_CNN_Feature_Extractor")
    net.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return net


def build_gru_temporal_modeling(input_shape, num_classes, learning_rate: float = 1e-3) -> keras.Model:
    """1D CNN for feature extraction + GRU for temporal modeling."""
    # Hyperparameters (kept inside this function only)
    conv_filters = (32, 64)
    kernel_sizes = (5, 3)
    rnn_units = 64
    dense_units = 64
    dropout = 0.3

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv1D(conv_filters[0], kernel_size=kernel_sizes[0], activation="relu")(inputs)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Conv1D(conv_filters[1], kernel_size=kernel_sizes[1], activation="relu")(x)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.GRU(rnn_units)(x)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    net = keras.Model(inputs, outputs, name="GRU_temporal_modeling")
    net.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return net
