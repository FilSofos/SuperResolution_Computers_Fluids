import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Activation
from tensorflow.keras.layers import MaxPooling2D,UpSampling2D,ZeroPadding2D,Cropping2D,AveragePooling2D, Dense, Add
from tensorflow.keras.models import Model
import os
import numpy as np
import random

##################################################################

# Define the scale-down factor
s2i=4

# Define the paths to the image folder
# Select train and test images
# One can also use random data split

data_dir = 'sudexp'
images_train = [f for f in os.listdir(data_dir) if f.endswith('.png')][:350]
images_test = [f for f in os.listdir(data_dir) if f.endswith('.png')][350:]


# Function that pre-process images 
def image_in(images_train,data_dir):
    fine_images_train=[]
    i=0
    for image_filename in images_train:
        image_path = os.path.join(data_dir, image_filename)
        image = cv2.imread(image_path)
        norm_image = image / 255.0
        fine_images_train.append(norm_image)

    height, width, channels = norm_image.shape
    print(f'Image Width: {width} pixels')
    print(f'Image Height: {height} pixels')
    print(f'Number of Channels: {channels}')
    fine_images_train_np = np.array(fine_images_train)
    return fine_images_train_np

fine_images_train_np=image_in(images_train,data_dir)
fine_images_test_np=image_in(images_test,data_dir)

# Function that creates blurred image counterparts for CNN training 
def image_blur(fine_images_train_np,s2i):
    
    # Change the dimension of the input images according to the application
    dim1 = (int(2296/s2i),int(325/s2i))
    coarse2_images_train = []
    for fine_image in fine_images_train_np:
        coarse2_image = cv2.resize(fine_image, dim1, interpolation=cv2.INTER_LINEAR)
        coarse2_images_train.append(coarse2_image)
    coarse2_images_train_np = np.array(coarse2_images_train)
    
    #First reduce dimensions, then linear intererpolate
    blurred_images_train = []
    dim = (2296,325)
    for fine_image in coarse2_images_train_np:
        blurred_image = cv2.resize(fine_image, dim, interpolation=cv2.INTER_LINEAR)
        blurred_images_train.append(blurred_image)
    blurred_images_train_np = np.array(blurred_images_train)
    return blurred_images_train_np

blurred_images_train_np=image_blur(fine_images_train_np,s2i)
blurred_images_test_np=image_blur(fine_images_test_np,s2i)

print('Create blurred_images_train: ',len(blurred_images_train_np))
print('Create blurred_images_test: ',len(blurred_images_test_np))

########################################################################################
# This is the basic model that performs super resolution
def build_model_resblur():
    # Coarse Input
    coarse_input = Input(shape=(325,2296, 3))
    coarse_x = Conv2D(32, (3, 3), activation='relu', padding='same')(coarse_input)
    coarse_x = Conv2D(32, (3, 3), activation='relu', padding='same')(coarse_x)
    coarse_x_residual = Conv2D(32, (1, 1), activation='relu', padding='same')(coarse_input)  # Add 1x1 conv for residual
    merged= Add()([coarse_x_residual, coarse_x])
    
    # Decoder (transpose convolution)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(merged)
    x = Conv2DTranspose(3, (3, 3), activation='relu', padding='same')(x)
    model = Model(inputs=coarse_input, outputs=x)
    return model

# Build the model
model = build_model_resblur()
model.summary()
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
#######################################################################

import time
# Start timing
start_time = time.time()

# Define the EarlyStopping callback
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Add the EarlyStopping callback to the list of callbacks when calling model.fit
history = model.fit(
    blurred_images_train_np,
    fine_images_train_np,
    epochs=100,
    batch_size=1,
    validation_data=(blurred_images_test_np, [fine_images_test_np]),
    verbose=1,
    shuffle=True,
    callbacks=[early_stopping]  # Add the EarlyStopping callback here
)

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")



