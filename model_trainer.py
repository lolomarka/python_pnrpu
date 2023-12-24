import tensorflow as tf
import os
# 0 - ART
# 1 - EOS
# 2 - LIN
# 3 - LYM
# 4 - MAC
# 5 - MON
# 6 - NEU
# 7 - NRBC
# 8 - PLA
# 9 - RBC
# image - 249x249 .jpg

TRAINING_DIR = "./_dataset/train/"
VALIDATION_DIR = "./_dataset/val/"
training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(249,249),
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(249,249),
    class_mode='categorical'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(249,249,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') 
])

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_generator, epochs=25,
                              validation_data = validation_generator,
                              verbose = 1, callbacks = [cp_callback])

model.save('model.keras')