import tensorflow as tf
from tensorflow.keras import layers


class TimeGAN(tf.keras.Model):
    def __init__(self, seq_len, feature_dim, hidden_dim, num_layers):
        super(TimeGAN, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding network (encoder)
        self.embedder = tf.keras.Sequential([
            layers.Input(shape=(seq_len, feature_dim)),
            *[layers.GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)],
            layers.TimeDistributed(layers.Dense(hidden_dim, activation='sigmoid'))
        ])

        # Recovery network (decoder)
        self.recovery = tf.keras.Sequential([
            layers.Input(shape=(seq_len, hidden_dim)),
            *[layers.GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)],
            layers.TimeDistributed(layers.Dense(feature_dim))
        ])

        # Generator (latent space)
        self.generator = tf.keras.Sequential([
            layers.Input(shape=(seq_len, hidden_dim)),
            *[layers.GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)],
            layers.TimeDistributed(layers.Dense(hidden_dim))
        ])

        # Supervisor (predict next in latent)
        self.supervisor = tf.keras.Sequential([
            layers.Input(shape=(seq_len, hidden_dim)),
            *[layers.GRU(hidden_dim, return_sequences=True) for _ in range(num_layers - 1)],
            layers.TimeDistributed(layers.Dense(hidden_dim))
        ])

        # Discriminator (real vs fake in latent space)
        self.discriminator = tf.keras.Sequential([
            layers.Input(shape=(seq_len, hidden_dim)),
            *[layers.GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)],
            layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))
        ])

    def call(self, x, training=False):
        # Basic forward pass for embedder → recovery → reconstruction
        h = self.embedder(x)
        x_tilde = self.recovery(h)
        return x_tilde

    def generate_latent(self, z):
        # Generate latent time series from random noise
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        return h_hat

    def generate_data(self, z):
        h_hat = self.generate_latent(z)
        x_hat = self.recovery(h_hat)
        return x_hat
