from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


batch_size = 32
image_size = (224, 224)
input_shape = (224, 224, 3)

train_data_dir = "input/processed_data/train"
val_data_dir = "input/processed_data/val"
test_data_dir = "input/processed_data/test"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    val_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = train_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

base_model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)

for layer in base_model.layers:
    layer.trainable = True

x = base_model.output
x = Flatten()(x)
x = Dense(512, kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(16, kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history1 = model.fit_generator(
    generator=train_generator,
    epochs=10,
    shuffle=True,
    verbose=1
)

for layer in model.layers:
    layer.trainable = False

for layer in model.layers[:50]:
    layer.trainable = True

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit_generator(
    generator=train_generator,
    epochs=50,
    shuffle=True,
    verbose=1
)

# Save model
model.save('ResNet50_retrained.h5')

# Plot the training graph
acc = history1.history['acc'] + history2.history['acc']
val_acc = history1.history['val_loss'] + history2.history['val_loss']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
epochs = range(len(acc))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')
axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
axes[0].set_title('Training and Validation Accuracy')
axes[0].legend(loc='best')

axes[1].plot(epochs, loss, 'r-', label='Training Loss')
axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
axes[1].set_title('Training and Validation Loss')
axes[1].legend(loc='best')

plt.show()

# Final test accuracy
_, test_acc = model.evaluate(test_generator)
print('Test accuracy', test_acc)
