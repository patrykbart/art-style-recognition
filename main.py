from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

model = load_model('ResNet50_retrained.h5')
test_data_dir = "input/processed_data/test"

batch_size = 32
image_size = (224, 224)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

test_generator = train_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Final test accuracy
_, test_acc = model.evaluate(test_generator)
print('Test accuracy', test_acc)