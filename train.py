from discriminater import make_discriminator_model
from generator import make_generator_model


import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython import display
from PIL import Image
import glob
import cv2 as cv

filelist = './images/*.PNG' #replace with your image location
pa="./sav/" #directory tosave images
ex=".png" #extension for images
checkpoint_dir = './training_checkpoints' #sirectory to save checkpoints they save every 200 epochs
all_images = []
for index, filename in enumerate(glob.glob(filelist)):
    image = cv.imread(filename)
    all_images.append(image)

X = np.array(all_images)
train_images = X.reshape(X.shape[0], 64 , 64 , 3).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]


BUFFER_SIZE =500
BATCH_SIZE = 128
inputshape=10
#batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



generator = make_generator_model(inputshape)
discriminator = make_discriminator_model()  


print(generator.summary())
print(discriminator.summary())


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)



checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)



EPOCHS = 10000
noise_dim = inputshape
num_examples_to_generate = 1

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    #noise = tf.random.normal([BATCH_SIZE, noise_dim])
    noise = np.random.uniform(-1., 1., size=[BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        
        for a in dataset:
            train_step(a)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,epoch + 1,seed)

        # Save the model every 200 epochs
        if (epoch + 1) % 200 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    #Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8,8))
    plt.imshow(predictions[0])
    #plt.imshow(predictions[0])
    #for i in range(predictions.shape[0]):
    #    plt.subplot(4, 4, i+1)
     #   plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
    #    plt.axis('off')

    na=epoch

    plt.savefig(pa+str(epoch)+ex)
    #plt.show()

train(train_dataset, EPOCHS)