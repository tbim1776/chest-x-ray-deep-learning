#The program creates a model that will learn on the training data, then saves the model. 
#Training data contains 26600 image, validation and test data contains 5700-5700 images
# 50% of the images are diagnosed with Infiltration, and 50% are healthy 

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Define constants used
BATCH_SIZE = 32
TARGET_SIZE = 256
LEARNING_RATE = 0.000001
STEPS_PER_EPOCH = 831
VALIDATION_AND_TEST_STEPS = 178
EPOCHS = 7

#Creating data generators
train_datagen = ImageDataGenerator( 
        rotation_range=20,
        zoom_range=0.1,
        rescale=1./255
        )

validation_datagen = ImageDataGenerator( 
        rescale=1./255
        )

test_datagen = ImageDataGenerator( 
        rescale=1./255
        )


#This will load the data
train_it = train_datagen.flow_from_directory('./train/', 
        class_mode='sparse',
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE)

validation_it = test_datagen.flow_from_directory(
        './validation',
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse')

test_it = test_datagen.flow_from_directory('./test/', 
        class_mode='sparse', 
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE)

#Creating a sequential model and adding layers
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(TARGET_SIZE, TARGET_SIZE, 3)))

model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(trainable=True))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(trainable=True))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(trainable=True))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.ZeroPadding2D(padding=(1, 1), data_format=None))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(trainable=True))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2))

#Compiling model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./checkpoints/model_weights', 
    verbose=1, 
    save_weights_only=True,
    save_best_only=True)

#Training
model.fit_generator(train_it,
                steps_per_epoch=STEPS_PER_EPOCH, 
                validation_data=validation_it, 
                validation_steps=VALIDATION_AND_TEST_STEPS, 
                epochs = EPOCHS,
                callbacks=[cp_callback])

#Load best weights
model.load_weights('./checkpoints/model_weights')

#Checking the model on the test data
loss, test_acc = model.evaluate_generator(test_it, steps=VALIDATION_AND_TEST_STEPS)

#Saving model
model.save('created_model')

#Printing loss and validation
print("loss: " + str(loss))
print("test accuracy: " + str(test_acc*100) + "%")
