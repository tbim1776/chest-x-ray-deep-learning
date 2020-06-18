import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

from skimage import transform

#class_names=['Infiltration', 'No Finding']

#The gui class
class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Image viewer")
        self.minsize(640, 400)

        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)

        self.button()


    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.button.grid(column = 1, row = 1)


    def fileDialog(self):

        self.filename = filedialog.askopenfilename(initialdir =  "~/Desktop/", title = "Select A File")
        self.label = ttk.Label(self.labelFrame, text = "Select a file")
        self.label.grid(column = 0, row = 2)
        self.label.configure(text = self.filename)

        img = Image.open(self.filename)
        img = img.resize((256,256))
        photo = ImageTk.PhotoImage(img)

        self.label2 = Label(image=photo)
        self.label2.image = photo 
        self.label2.grid(column=0, row=3)

        image = load(self.filename)
        pr = probability_model.predict(image)
        #print(pr)
        classes = np.argmax(pr, axis = 1)

        labels = (train_it.class_indices)
        labels = dict((v,k) for k,v in labels.items())    

        self.label3 = Label(text = [labels[k] for k in classes])
        self.label3.grid(column=0, row=5)

#loads and augments the selected image
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (256, 256, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

train_datagen = ImageDataGenerator()

train_it = train_datagen.flow_from_directory('./train/', 
        class_mode='sparse',
        target_size=(256, 256),
        batch_size=32)

#loading the model created by create_model.py, changing it to be able to predict

model = load_model('created_model')

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) 

for layer in probability_model.layers: layer.trainable = False

#opening the gui
root = Root()
root.mainloop()

