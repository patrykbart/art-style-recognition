import os
import cv2
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import *
from tensorflow.keras.models import load_model

plt.figure(figsize=(12, 6), dpi=80)
model = load_model('ResNet50_retrained.h5')

image_size = (224, 224)

targets = {0: 'Expressionism', 1: 'Impressionism', 2: 'Realism', 3: 'Renaissance', 4: 'Romanticism'}

style = random.choice(list(targets.values()))
files = os.listdir('input/processed_data/test/' + style)

path_to_img = 'input/processed_data/test/' + style + '/' + random.choice(list(files))
print(path_to_img)

img = cv2.imread(path_to_img)
img = cv2.resize(img, image_size, 3)
img = np.array(img).astype(np.float32)/255.0
img = np.expand_dims(img, axis=0)

output = model.predict(img)[0]
output = [round(prob * 100, 2) for prob in output]

plt.barh(list(targets.values()), output)
for i, prob in enumerate(output):
    plt.text(prob + 0.1, i - 0.05, str(prob))

plt.subplots_adjust(left=0.1)
plt.xlabel('%')

plt.savefig('plot.png')
plt.show()
