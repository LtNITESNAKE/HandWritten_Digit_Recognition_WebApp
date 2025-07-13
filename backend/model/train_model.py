import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Create data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.GaussianNoise(0.1)
])

# Build an improved CNN model with residual connections
def build_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = data_augmentation(inputs, training=True)
    
    # First Convolutional Block with residual connection
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    # Add residual connection
    conv1 = layers.Add()([x, conv1]) if x.shape[-1] == conv1.shape[-1] else conv1
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    drop1 = layers.Dropout(0.25)(pool1)

    # Second Convolutional Block with residual connection
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    # Add residual connection
    shortcut = layers.Conv2D(64, (1, 1))(drop1)
    conv2 = layers.Add()([shortcut, conv2])
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    drop2 = layers.Dropout(0.25)(pool2)

    # Third Convolutional Block with residual connection
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(drop2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    # Add residual connection
    shortcut = layers.Conv2D(128, (1, 1))(drop2)
    conv3 = layers.Add()([shortcut, conv3])
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    drop3 = layers.Dropout(0.25)(pool3)

    # Dense Layers
    flat = layers.Flatten()(drop3)
    dense1 = layers.Dense(512, activation='relu')(flat)
    bn1 = layers.BatchNormalization()(dense1)
    drop4 = layers.Dropout(0.5)(bn1)
    dense2 = layers.Dense(256, activation='relu')(drop4)
    bn2 = layers.BatchNormalization()(dense2)
    drop5 = layers.Dropout(0.5)(bn2)
    outputs = layers.Dense(10, activation='softmax')(drop5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
model = build_model()

# Setup learning rate parameters
initial_learning_rate = 0.001
min_learning_rate = 0.0001

# Use gradient clipping to prevent exploding gradients
optimizer = tf.keras.optimizers.Adam(
    learning_rate=initial_learning_rate,
    clipnorm=1.0  # Gradient clipping
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Create callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'digit_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Reduce learning rate on plateau - this will adaptively decrease the learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=min_learning_rate,
    verbose=1  # Added verbose to see when the learning rate changes
)

# Train the model with augmented data
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=4,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Evaluate the final model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'\nFinal test accuracy: {test_accuracy*100:.2f}%')

# Save training history
np.save('training_history.npy', history.history)

# Save the final model state
model.save('final_digit_model.h5')

# Save the model architecture as JSON
model_json = model.to_json()
with open("model_architecture.json", "w") as json_file:
    json_file.write(model_json)
print("\nModel and architecture have been saved successfully.")
