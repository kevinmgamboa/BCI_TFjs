"""
This file contain models implemented for the project
----------------------------

"""

# -----------------------------------------------------------------------------
#                           Libraries Needed
# -----------------------------------------------------------------------------
from helpers_and_functions import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class simple_cnn():
    def __init__(self, param):
        # Initializing model
        self.model = None
        # parameter
        self.parameters = param
        # # Building model structure
        # self.structure()
        # # Compiling model
        # self.compile()

    def build_model_structure(self, in_shape):
        # Adds batch size = 1
        in_shape = (1,) + in_shape
        num_filters = 20
        kernel_size = 3
        out_size = 1

        self.model = tf.keras.Sequential([
            layers.Conv2D(num_filters, kernel_size, padding='same', activation='relu',
                          kernel_initializer='he_normal', input_shape=in_shape),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(out_size)
        ])


    def compile(self):
        self.model.compile(optimizer=keras.optimizers.Adam(self.parameters['lr']),
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=keras.metrics.BinaryAccuracy(name='accuracy'))


# -----------------------------------------------------------------------------
#                           Multi-Branch Model
# -----------------------------------------------------------------------------
class multi_output_feature_model:
    def __init__(self, param, input_shape):
        # initializing model object
        self.model = None
        # initializing parameters
        self.parameters = param
        # builds the model
        self.build_model((1,)+input_shape)  # adds extra dimension to in_shape


    def spectrogram_input_branch(self, x_in):
        x = layers.Conv2D(self.parameters['num_filters'], self.parameters['kernel_size'],
                          padding='same', activation='relu', name='spec_conv_1')(x_in)
        x = layers.Flatten(name='spec_flatten')(x)
        x = layers.Dense(self.parameters['dense_units'], activation='relu', name='spec_dense_1')(x)
        x = layers.Dense(self.parameters['out_size'], name='spec_dense_2')(x)

        return x


    def hilbert_transform_branch(self, x_in):
        x = layers.Conv2D(self.parameters['num_filters'], self.parameters['kernel_size'],
                          padding='same', activation='relu', name='hilbert_conv_1')(x_in)
        x = layers.Flatten(name='hilbert_flatten')(x)
        x = layers.Dense(self.parameters['dense_units'], activation='relu', name='hilbert_dense_1')(x)
        x = layers.Dense(self.parameters['out_size'], name='hilbert_dense_2')(x)

        return x

    def build_model(self, input_shape):
        # creates the model input
        x_in = keras.Input(input_shape)
        # builds the model
        self.model = keras.models.Model(inputs=x_in,
                                        outputs=[self.spectrogram_input_branch(x_in), self.hilbert_transform_branch(x_in)],
                                        name='multi_input_feature_model')

# %%
# -----------------------------------------------------------------------------
#          Multi-Branch Model (https://keras.io/guides/functional_api/)
# -----------------------------------------------------------------------------
class multi_input_feature_model:
    def __init__(self, param, input_1, input_2):
        # initializing model object
        self.model = None
        # initializing parameters
        self.parameters = param
        # builds the model
        self.build_model_structure((1,) + input_1, (1,) + input_2)  # adds extra dimension to in_shape

    def spectrogram_input_branch(self, x_in):
        # creating model body structure
        x = layers.Conv2D(self.parameters['num_filters'], self.parameters['kernel_size'],
                          padding='same', activation='relu')(x_in)
        x = layers.Flatten()(x)
        x = layers.Dense(self.parameters['dense_units'], activation='relu')(x)
        x = layers.Dense(self.parameters['out_size'])(x)
        # defining model output
        y = keras.models.Model(inputs=x_in, outputs=x)

        return y

    def hilbert_transform_branch(self, x_in):
        # creating model body structure
        x = layers.Conv2D(self.parameters['num_filters'], self.parameters['kernel_size'],
                          padding='same', activation='relu')(x_in)
        x = layers.Flatten()(x)
        x = layers.Dense(self.parameters['dense_units'], activation='relu')(x)
        x = layers.Dense(self.parameters['out_size'])(x)
        # defines the output
        y = keras.models.Model(inputs=x_in, outputs=x)

        return y

    def build_model_structure(self, input_1, input_2):
        # creates input for spectrogram and for hilbert transform
        spec_input, hilb_input = keras.Input(input_1, name='spectrogram_input'), keras.Input(input_2,
                                                                                             name='hilbert_input')
        # creates spectrogram and hilbert transform branches
        spec_b, hilb_b = self.spectrogram_input_branch(spec_input), self.spectrogram_input_branch(hilb_input)
        # combining branches outputs via concatenation
        comb_out = layers.concatenate([spec_b.output, hilb_b.output])
        # adding final extra layers
        conscious_pred = layers.Dense(self.parameters['out_size'], name='final_out')(comb_out)
        # builds the model
        self.model = keras.models.Model(inputs=[spec_b.input, hilb_b.input],
                                        outputs=conscious_pred,
                                        name='multi_input_feature_model')

    def compile(self):
        self.model.compile(optimizer=keras.optimizers.Adam(self.parameters['lr']),
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=keras.metrics.BinaryAccuracy(name='accuracy'))

# %%
# -----------------------------------------------------------------------------
#          VAE (https://keras.io/guides/functional_api/)
# -----------------------------------------------------------------------------
class CVACmodel():
    def __init__(self, param):
        # Initializing model
        self.model = None
        # parameter
        self.parameters = param
        # # Building model structure
        # self.structure()
        # # Compiling model
        # self.compile()

    def build_model_structure(self):
        # Creates encoder input
        x_in = keras.Input(shape=(1,) + self.parameters['input_shape'], name='encoder_in')
        # Defining convolutions
        x = layers.Conv2D(self.parameters['num_filters'], self.parameters['kernel_size'],
                          padding='same', activation='relu')(x_in)
        x = layers.Flatten()(x)
        # Defining multilayer perceptron
        x = layers.Dense(self.parameters['dense_units'], activation='relu')(x)
        x = layers.Dense(self.parameters['dense_units'], activation='relu')(x)

        z_mu = layers.Dense(self.parameters['latent_dimension'], name='latent_mu')(x)
        z_sigma = layers.Dense(self.parameters['latent_dimension'], name='latent_sigma')(x)

        z = layers.Lambda(self.sample_z, output_shape=(self.parameters['latent_dimension'],), name='z')([z_mu, z_sigma])

        # adding final extra layers
        y_out = layers.Dense(self.parameters['out_size'], name='final_out')(z)

        # builds the model
        self.model = keras.models.Model(inputs=x_in,
                                        outputs=y_out,
                                        name='VCNN')

    def sample_z(args):
        z_mu, z_sigma = args
        eps = keras.backend.random_normal(shape=(keras.backend.shape(z_mu)[0], keras.backend.int_shape(z_mu)[1]))
        return z_mu + keras.backend.exp(z_sigma / 2) * eps

    def compile(self):

        self.model.compile(optimizer=keras.optimizers.Adam(self.parameters['lr']),
                           loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=keras.metrics.BinaryAccuracy(name='accuracy'))


# class CVAE(tf.keras.Model):
#   """Convolutional variational autoencoder."""
#
#   def __init__(self, latent_dim):
#     super(CVAE, self).__init__()
#     self.latent_dim = latent_dim
#     self.encoder = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
#             tf.keras.layers.Conv2D(
#                 filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
#             tf.keras.layers.Conv2D(
#                 filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
#             tf.keras.layers.Flatten(),
#             # No activation
#             tf.keras.layers.Dense(latent_dim + latent_dim),
#         ]
#     )
#
#     self.decoder = tf.keras.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#             tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
#             tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=64, kernel_size=3, strides=2, padding='same',
#                 activation='relu'),
#             tf.keras.layers.Conv2DTranspose(
#                 filters=32, kernel_size=3, strides=2, padding='same',
#                 activation='relu'),
#             # No activation
#             tf.keras.layers.Conv2DTranspose(
#                 filters=1, kernel_size=3, strides=1, padding='same'),
#         ]
#     )
#
#   @tf.function
#   def sample(self, eps=None):
#     if eps is None:
#       eps = tf.random.normal(shape=(100, self.latent_dim))
#     return self.decode(eps, apply_sigmoid=True)
#
#   def encode(self, x):
#     mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
#     return mean, logvar
#
#   def reparameterize(self, mean, logvar):
#     eps = tf.random.normal(shape=mean.shape)
#     return eps * tf.exp(logvar * .5) + mean
#
#   def decode(self, z, apply_sigmoid=False):
#     logits = self.decoder(z)
#     if apply_sigmoid:
#       probs = tf.sigmoid(logits)
#       return probs
#     return logits
