import tensorflow as tf

def build_omnifold_classifier(n_features: int) -> tf.keras.Model:
    """
    Build a neural network model for the Omnifold classifier.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,), name="features"),

        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Dense(1, activation="sigmoid", name="classification"),
    ], name="omnifold_classifier")

    return model