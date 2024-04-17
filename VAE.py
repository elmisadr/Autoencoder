import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


# create a custom sampling layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_sigma = tf.exp(0.5 * z_log_var)

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + z_sigma * epsilon

def build_vae(input_dim, filters, latent_dim):
    chan_dim = -1
    ## --- encoder
    inputs = layers.Input(shape=input_dim)
    # loop over the number of filters
    x = inputs
    for f in filters:
        x = layers.Conv2D(f, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization(axis=chan_dim)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

    # get the image dimension of the last convolutional layer
    conv_img_dim = K.int_shape(x)

    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)

    # latent layers
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    # sampling layer
    z = Sampling()([z_mean, z_log_var])

    # build the encoder model
    encoder = Model(inputs, z, name="encoder")

    ## --- decoder
    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(np.prod(conv_img_dim[1:]), activation="relu")(latent_inputs)
    x = layers.Reshape((conv_img_dim[1], conv_img_dim[2], conv_img_dim[3]))(x)

    # loop over our number of filters, this time in reverse order
    for f in filters[::-1]:
        x = layers.Conv2DTranspose(f, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization(axis=chan_dim)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
    # apply a single CONV_TRANSPOSE layer used to recover the
    # original depth of the image
    depth = input_dim[2]
    x = layers.Conv2DTranspose(depth, kernel_size=3, padding="same")(x)
    outputs = layers.Activation("sigmoid")(x)
    
    # build the decoder model
    decoder = Model(latent_inputs, outputs, name="decoder")

    # build VAE, which is the encoder + decoder
    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs, name="vae")

    # compute custom VAE loss
    reconstruction_loss = K.mean(K.sum(keras.losses.binary_crossentropy(inputs, outputs), 
                                       axis=(1, 2)))
    z_sigma = K.exp(0.5 * z_log_var)
    kl_loss = 1 + K.log(z_sigma) - K.square(z_mean) - z_sigma
    kl_loss = -0.5 * K.sum(kl_loss, axis=1)
    kl_loss = K.mean(kl_loss)

    vae_loss = reconstruction_loss+kl_loss

    # compile the model
    vae.add_loss(vae_loss)
    vae.compile(optimizer=keras.optimizers.Adam(lr=1e-3))

    # return a 3-tuple of the encoder, decoder, and vae
    return (encoder, decoder, vae)

from tensorflow.keras.datasets import mnist
import numpy as np

# load the MNIST dataset
((trainX, trainY), (testX, _)) = mnist.load_data()
# add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# construct our VAE
print("Building VAE...")
input_dim = trainX.shape[1:]
filters = (32, 64)
latent_dim = 2
(encoder, decoder, vae) = build_vae(input_dim, filters, latent_dim=latent_dim)

encoder.summary()
decoder.summary()
vae.summary()
# initialize the number of epochs to train for and batch size
EPOCHS = 25
BS = 32

H = vae.fit(
	trainX, trainX,
	validation_data=(testX, testX),
	epochs=EPOCHS,
	batch_size=BS)
import matplotlib.pyplot as plt

plt.figure()
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
fig_plot = plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()
def plot_latent_space(decoder, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.gca()
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    ax.grid(False)
    plt.show()


plot_latent_space(decoder)
def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z[:, 0], z[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z[:, 0], z[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

plot_label_clusters(encoder, trainX, trainY)
 # display a 2D plot of the digit classes in the latent space
    z = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z[:, 0], z[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

from PIL import Image

decoded = vae.predict(testX)
outputs = np.empty((0,testX[0].shape[1]*2, testX[0].shape[2]))

# number of samples to visualize when decoding
samples_to_visualize = 8

# loop over our number of output samples
for i in range(0, samples_to_visualize):
  # grab the original image and reconstructed image
  original = (testX[i] * 255).astype("uint8")  
  recon = (decoded[i] * 255).astype("uint8")
  # stack the original and reconstructed image side-by-side
  output = np.hstack([original, recon])
  outputs = np.vstack([outputs, output])

outputs = np.reshape(outputs,(outputs.shape[0],outputs.shape[1]))
img = Image.fromarray(outputs)
fig_images = plt.figure()
plt.axis('off')
plt.imshow(img)
plt.title("Original and reconstructed images side-by-side")
plt.show()
