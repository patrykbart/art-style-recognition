from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model


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

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')

history1 = model.fit_generator(
    generator=train_generator,
    epochs=10,
    shuffle=True,
    verbose=1,
    callbacks=[reduce_lr]
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
    verbose=1,
    callbacks=[reduce_lr, early_stop]
)

# Save model
model.save('ResNet50_retrained.h5')
print('Model saved')
