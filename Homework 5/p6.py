# toc
print('Importing stuff')
import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import certifi
import numpy as np
import seaborn as sns
import cv2

plt.style.use('maroon_pad.mplstyle')

from sklearn.metrics import confusion_matrix, accuracy_score

# keras/tensorflow
from keras.api.datasets import cifar10
from keras.api.models import Sequential, load_model
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, GlobalAveragePooling2D, RandomFlip, RandomTranslation, RandomRotation, RandomZoom
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from keras.api.applications import MobileNetV2
from keras.api.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

# If you have a raw python installation, you have to set the SSL_CERT_FILE environment variable
os.environ['SSL_CERT_FILE'] = certifi.where()

tf.config.set_visible_devices([], 'GPU')
# plt.style.use('../maroon_ipynb.mplstyle')

# load the dataset
print('Loading data')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize the data
x_train = x_train/255
x_test = x_test/255

class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Preprocess CIFAR-10 images for MobileNetV2:
# Our x_train and x_test are normalized between 0 and 1, but MobileNetV2 expects inputs in a different scale.
# First, rescale back to [0, 255], resize to 224x224, then use the MobileNetV2 preprocessing.
x_train_rescaled = (x_train*255).astype('float32')
x_test_rescaled = (x_test*255).astype('float32')

def resize_images(images, size=(224, 224)):
    resized = np.zeros((images.shape[0], size[0], size[1], images.shape[3]), dtype=np.float32)
    for im in range(images.shape[0]):
        resized[im] = cv2.resize(images[im], size)
    return resized

x_train_resized = resize_images(x_train_rescaled)
x_test_resized = resize_images(x_test_rescaled)

x_train_preprocessed = preprocess_input(x_train_resized)
x_test_preprocessed = preprocess_input(x_test_resized)
print('Processed images')

# Load the pretrained MobileNetV2 model without its top, and freeze its layers.
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

# Build our new model by stacking our dense layers on top of the frozen base.
p6_model = Sequential([
    RandomFlip(mode='horizontal'),
    RandomTranslation(0.2, 0.2),
    RandomRotation(0.2),
    RandomZoom(0.2),
    base_model,
    GlobalAveragePooling2D(),
    Dense(2048, activation='relu'),  # optimized dense layer
    Dense(10, activation='softmax')
])

# Compile the model.
p6_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Create callback
checkpoint_callback = ModelCheckpoint(
    'best_model_p6.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=True
)

# Train the model.
print('Training model')
history_p6 = p6_model.fit(
    x_train_preprocessed, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(x_test_preprocessed, y_test),
    callbacks=[checkpoint_callback],
    verbose=True
)

# Get accuracy results
train_accuracy = history_p6.history['accuracy']
val_accuracy = history_p6.history['val_accuracy']
epochs = np.array(history_p6.epoch) + 1
best_accuracy = max(val_accuracy)
best_accuracy_epoch = val_accuracy.index(best_accuracy) + 1

# Plot the training and validation accuracy
fig, ax = plt.subplots()
line = ax.plot(epochs, train_accuracy, label=f'Training Accuracy', marker='.', zorder=2)[0]
ax.plot(epochs, val_accuracy, label=f'Validation Accuracy', marker='.', ls='--', color=line.get_color(), zorder=2)
ax.scatter(best_accuracy_epoch, best_accuracy, marker='x', color=line.get_color(), zorder=3, label=fr'Best Val Accuracy ({best_accuracy*100:.2f})')
ax.set_title('Training and Validation Accuracy')
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.legend()
plt.savefig('p6.png')

# Making the confusion matrix
p6_best_model = load_model('best_model_p6.keras')  # Load the best model
y_pred = p6_best_model.predict(x_test_preprocessed, verbose=False)  # Get predictions on the test set

# Convert predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)  # finding the highest probability
cm = confusion_matrix(y_test.flatten(), y_pred_labels)

# Plot confusion matrix
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=45)
ax.grid(False)
plt.savefig('p6_cm.png')

# Accuracy score
accuracy = accuracy_score(y_test.flatten(), y_pred_labels)
print(f'Accuracy of the best model on the test set: {accuracy:.3%}')
