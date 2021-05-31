from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model


batch_size = 64
image_size = (224, 224)
input_shape = (224, 224, 3)

train_data_dir = "input/processed_data/train"
val_data_dir = "input/processed_data/val"

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

base_model = MobileNet(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)

output = Flatten()(base_model.output)
output = Dense(train_generator.num_classes, activation='softmax')(output)

model = Model(inputs=base_model.inputs, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
    generator=train_generator,
    epochs=10,
    shuffle=True,
    verbose=1,
    validation_data=val_generator
)

model.save('MobileNet.h5')
print('Model saved')
