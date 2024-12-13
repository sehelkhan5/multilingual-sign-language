# # Importing the Libraries
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
# import os

# # Set the GPU device to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# # Check TensorFlow version
# print("TensorFlow version:", tf.__version__)

# ### Part 1 - Data Preprocessing

# #### Generating images for the Training set
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                     shear_range=0.2,
#                                     zoom_range=0.2,
#                                     horizontal_flip=True)

# #### Generating images for the Test set
# test_datagen = ImageDataGenerator(rescale=1./255)

# ### Creating the Training set
# training_set = train_datagen.flow_from_directory(
#     'tkdi/trainingData',
#     target_size=(128, 128),
#     batch_size=10,
#     color_mode='grayscale',
#     class_mode='categorical'
# )

# ### Creating the Test set
# test_set = test_datagen.flow_from_directory(
#     'tkdi/testingData',
#     target_size=(128, 128),
#     batch_size=10,
#     color_mode='grayscale',
#     class_mode='categorical'
# )

# ### Part 2 - Building the CNN

# #### Initializing the CNN
# classifier = tf.keras.models.Sequential()

# #### Step 1 - Convolution
# classifier.add(tf.keras.layers.Conv2D(
#     filters=32,
#     kernel_size=3,
#     padding="same",
#     activation="relu",
#     input_shape=[128, 128, 1]
# ))

# #### Step 2 - Pooling
# classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# #### Adding a second convolutional layer
# classifier.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
# classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# #### Step 3 - Flattening
# classifier.add(tf.keras.layers.Flatten())

# #### Step 4 - Full Connection
# classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
# classifier.add(tf.keras.layers.Dropout(0.40))
# classifier.add(tf.keras.layers.Dense(units=96, activation='relu'))
# classifier.add(tf.keras.layers.Dropout(0.40))
# classifier.add(tf.keras.layers.Dense(units=64, activation='relu'))
# classifier.add(tf.keras.layers.Dense(units=4, activation='softmax'))  # softmax for multi-class classification

# ### Part 3 - Training the CNN

# #### Compiling the CNN
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# #### Display the model summary
# classifier.summary()

# #### Training the CNN on the Training set and evaluating it on the Test set
# classifier.fit(training_set, epochs=5, validation_data=test_set)

# model_json = classifier.to_json()
# with open("testing/tkdi.json", 'w') as json_file:
#     json_file.write(model_json)

# classifier.save("testing/tkdi.h5")


# Importing necessary libraries
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # Set the GPU device to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# # Check TensorFlow version
# print("TensorFlow version:", tf.__version__)

# ### Part 1 - Data Preprocessing

# #### Generating images for the Training set with enhanced data augmentation
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                     shear_range=0.2,
#                                     zoom_range=0.2,
#                                     horizontal_flip=True,
#                                     rotation_range=20,
#                                     brightness_range=[0.8, 1.2])

# #### Generating images for the Test set
# test_datagen = ImageDataGenerator(rescale=1./255)

# ### Creating the Training set
# training_set = train_datagen.flow_from_directory(
#     'dataset/english/trainingData',
#     target_size=(128, 128),
#     batch_size=10,
#     color_mode='grayscale',
#     class_mode='categorical'
# )

# ### Creating the Test set
# test_set = test_datagen.flow_from_directory(
#     'dataset/english/testingData',
#     target_size=(128, 128),
#     batch_size=10,
#     color_mode='grayscale',
#     class_mode='categorical'
# )

# ### Part 2 - Building the CNN

# #### Initializing the CNN
# classifier = tf.keras.models.Sequential()

# #### Step 1 - Convolution
# classifier.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[128, 128, 1]))
# classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# #### Adding a second convolutional layer
# classifier.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
# classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# #### Adding a third convolutional layer
# classifier.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
# classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# #### Step 3 - Flattening
# classifier.add(tf.keras.layers.Flatten())

# #### Step 4 - Full Connection
# classifier.add(tf.keras.layers.Dense(units=256, activation='relu'))
# classifier.add(tf.keras.layers.Dropout(0.5))
# classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
# classifier.add(tf.keras.layers.Dropout(0.5))
# classifier.add(tf.keras.layers.Dense(units=len(training_set.class_indices), activation='softmax'))  # softmax for multi-class classification

# ### Part 3 - Training the CNN

# #### Compiling the CNN
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# #### Callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# #### Display the model summary
# classifier.summary()

# #### Training the CNN on the Training set and evaluating it on the Test set
# classifier.fit(training_set, epochs=10, validation_data=test_set, callbacks=[early_stopping, reduce_lr])

# # Save model structure and weights
# model_json = classifier.to_json()
# with open("testing_model/english.json", 'w') as json_file:
#     json_file.write(model_json)

# classifier.save("testing_model/english.h5")

# ### Part 4 - Evaluating the Model

# # Generate predictions for the test set
# test_set.reset()  # Resetting iterator to avoid errors
# predictions = classifier.predict(test_set, steps=len(test_set), verbose=1)
# predicted_classes = np.argmax(predictions, axis=1)
# true_classes = test_set.classes
# class_labels = list(test_set.class_indices.keys())

# # Generate classification report
# report = classification_report(true_classes, predicted_classes, target_names=class_labels, szero_division=1)
# print(report)

# # Confusion Matrix
# cm = confusion_matrix(true_classes, predicted_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set the GPU device to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

### Part 1 - Data Preprocessing

#### Generating images for the Training set with enhanced data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    rotation_range=20,
                                    brightness_range=[0.8, 1.2])

#### Generating images for the Test set
test_datagen = ImageDataGenerator(rescale=1./255)

### Creating the Training set
training_set = train_datagen.flow_from_directory(
    'dataset/german/trainingData',
    target_size=(128, 128),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical'
)

### Creating the Test set
test_set = test_datagen.flow_from_directory(
    'dataset/german/testingData',
    target_size=(128, 128),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical'
)

### Part 2 - Building the CNN

#### Initializing the CNN
classifier = tf.keras.models.Sequential()

#### Step 1 - Convolution
classifier.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[128, 128, 1]))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#### Adding a second convolutional layer
classifier.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#### Adding a third convolutional layer
classifier.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#### Step 3 - Flattening
classifier.add(tf.keras.layers.Flatten())

#### Step 4 - Full Connection
classifier.add(tf.keras.layers.Dense(units=256, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.5))
classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.5))
classifier.add(tf.keras.layers.Dense(units=len(training_set.class_indices), activation='softmax'))  # softmax for multi-class classification

### Part 3 - Training the CNN

#### Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#### Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

#### Display the model summary
classifier.summary()

#### Calculate Class Weights
# Get the class indices from the training set
class_indices = training_set.class_indices  # Dictionary mapping class names to indices

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(training_set.classes),
    y=training_set.classes
)

# Convert to dictionary format
class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)

#### Training the CNN on the Training set and evaluating it on the Test set
classifier.fit(
    training_set,
    epochs=10,
    validation_data=test_set,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict  # Add class weights here
)

# Save model structure and weights
model_json = classifier.to_json()
with open("testing_model/german.json", 'w') as json_file:
    json_file.write(model_json)

classifier.save("testing_model/german.h5")

### Part 4 - Evaluating the Model

# Generate predictions for the test set
test_set.reset()  # Resetting iterator to avoid errors
predictions = classifier.predict(test_set, steps=len(test_set), verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())

# Generate classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=1)
print(report)

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
