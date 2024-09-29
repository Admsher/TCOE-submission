#!/usr/bin/env python
# coding: utf-8

# ## Installing libraries and Dependencies
# 
# ### Uncomment and run the following code block to install all required dependencies

# In[3]:




# ## Importing libraries

# In[2]:


import cv2
import os
import numpy as np

import gradio as gr

import matplotlib.pyplot as plt

import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

from sklearn.metrics import classification_report


# ## Defining the `get_data` function to get data from the directory

# In[3]:


labels = ['red', 'black', 'geographic', 'normal', 'yellow'] # titles of subfolders
img_size = 120 # input image size




def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)


# ## Defining training and validation directories

# In[4]:


train = get_data('Dr_Tongue/data/train/')
val = get_data('Dr_Tongue/data/test')


# ## Generating the features and labels from the data and normalizing it

# In[5]:


x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)


# ## Data Augmentation

# In[6]:


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)   


# ## Model Definition

# In[7]:


model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(120,120,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(5, activation="softmax"))

model.summary()


# ## Model Compilation

# In[8]:


opt = Adam(learning_rate=0.0005)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])


# ## Model fitting

# In[9]:


history = model.fit(x_train,y_train, epochs = 25, validation_data = (x_val, y_val))


# ## Plotting Model's accuracy and loss w.r.t. training and validation set

# In[10]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(25)

# plt.figure(figsize=(20, 10))
# plt.subplot(2, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


# ## Making predictions

# In[11]:


# predictions = model.predict_step(x_val)
# # predictions = predictions.shape[0]
# print(len(y_val), len(predictions))

# print(classification_report(y_val, predictions, target_names = ['red (Class 0)','black (Class 1)', 'geographic (Class 2)', 'normal (Class 3)', 'yellow (Class 4)']))


# ## Predicting using local web interface at `http://127.0.0.1:7860/`

# In[12]:


# def predict_image(img):
#   img_4d=img.reshape(-1,120,120,3)
#   prediction=model.predict(img_4d)[0]
#   return {labels[i]: float(prediction[i]) for i in range(5)}

# image = gr.inputs.Image(shape=(120,120))
# label = gr.outputs.Label(num_top_classes=5)

# gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')

predictions = model.predict(x_val)  # Use model.predict instead of model.predict_step
predicted_classes = np.argmax(predictions, axis=1) 

print(f'Length of y_val: {len(y_val)}, Length of predictions: {len(predicted_classes)}')

# Print classification report
print(classification_report(y_val, predicted_classes, target_names=labels))

# Visualizing predictions
# def visualize_predictions(x_val, y_val, predicted_classes, num_images=5):
#     plt.figure(figsize=(15, 10))
#     for i in range(num_images):
#         plt.subplot(1, num_images, i + 1)
#         plt.imshow(x_val[i])  # Show the original image
#         plt.title(f'True: {labels[y_val[i]]}\nPred: {labels[predicted_classes[i]]}')
#         plt.axis('off')
#     plt.show()

# Call the visualization function
# visualize_predictions(x_val, y_val, predicted_classes)



# def predict_image(img_path):
#     img = cv2.imread(img_path)
#     img_resized = cv2.resize(img, (120, 120))  # Resize the image
#     img_4d = img_resized.reshape(-1, 120, 120, 3) / 255.0  # Normalize the image
#     prediction = model.predict(img_4d)[0]  # Make a prediction
#     return {labels[i]: float(prediction[i]) for i in range(len(labels))}

import pandas as pd

# Define a function to predict and save results
def predict_image_and_save(img_path, output_excel='predictions.xlsx'):
    img = cv2.imread(img_path)
    print(img)
    img_resized = cv2.resize(img, (120, 120))  # Resize the image
    img_4d = img_resized.reshape(-1, 120, 120, 3) / 255.0  # Normalize the image
    prediction = model.predict(img_4d)[0]  # Make a prediction

    # Prepare the prediction data for saving
    prediction_results = {labels[i]: float(prediction[i]) for i in range(len(labels))}
    prediction_results['Image Path'] = img_path  # Add the image path for reference

    # Create a DataFrame to hold the prediction results
    df = pd.DataFrame([prediction_results])

    # Save to Excel file
    # with pd.ExcelWriter(output_excel, mode='a', engine='openpyxl') as writer:
    df.to_excel(output_excel, index=False)

    return prediction_results

# Example usage
# result = predict_image_and_save('path/to/your/image.jpg')
# print(result)



# ## **Note:**
# 
# ### As can be observed, validation accuracy is lower than training accuracy, and validation loss is likewise higher than training loss. It's due to the fact that there's fewer data. By expanding the data set, accuracy may be improved and losses can be reduced.
