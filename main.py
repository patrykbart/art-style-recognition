import os
import cv2
import random
import tkinter as tk
import tkinter.font as font
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from keras.preprocessing.image import *
from tensorflow.keras.models import load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Application(tk.Frame):
    def __init__(self, master):
        self.master = master
        master.title('Painting style recognition')
        master.geometry('1205x450')

        self.image = None
        self.plot = None
        self.button = tk.Button(master, text='Load file', command=self.open_file)
        self.button.config(width=12, height=1)
        self.button['font'] = font.Font(size=16)
        self.button.pack(side=tk.BOTTOM)

        self.image_label = tk.Label()
        self.canvas = FigureCanvasTkAgg()

        self.model = load_model('ResNet50_retrained.h5')
        self.targets = {0: 'Expressionism', 1: 'Impressionism', 2: 'Realism', 3: 'Renaissance', 4: 'Romanticism'}

    def open_file(self):
        image = askopenfilename()

        self.image_label.destroy()
        self.canvas.get_tk_widget().destroy()

        self.load_img(image)
        self.predict(image)

    def load_img(self, image):
        image = Image.open(image).resize((400, 400))
        image = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(root, image=image)
        self.image_label.image = image
        self.image_label.pack(side=tk.LEFT)

    def predict(self, image):
        img = cv2.imread(image)
        img = cv2.resize(img, (224, 224), 3)
        img = np.array(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        output = self.model.predict(img)[0]
        output = [round(prob * 100, 2) for prob in output]

        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.subplots()

        ax.barh(list(self.targets.values()), output)
        for i, prob in enumerate(output):
            ax.text(prob + 0.1, i - 0.05, str(prob))

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel('%')
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT)

root = tk.Tk()
app = Application(root)
root.mainloop()
