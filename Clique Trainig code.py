from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob

# GPU memory configuration
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Image size
IMAGE_SIZE = [224, 224]

# Paths
train_path = '/Users/ishan/Downloads/PlantDiseasesDataset'
valid_path = '/Users/ishan/Downloads/PlantDiseasesDataset'

# Import InceptionV3 with pre-trained weights
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze existing weights
for layer in inception.layers:
    layer.trainable = False

# Get number of output classes
folders = glob('/Users/ishan/Downloads/PlantDiseasesDataset/*')

# Flatten output and add a dense layer for classification
x = Flatten()(inception.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Create the model
model = Model(inputs=inception.input, outputs=prediction)

# Compile the model
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the training and validation datasets
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Calculate steps per epoch and validation steps
steps_per_epoch = len(training_set)
validation_steps = len(test_set)

# Train the model for multiple epochs (increase the number of epochs)
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=10,  # Increase the number of epochs as needed
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)
