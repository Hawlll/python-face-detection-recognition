import os
import numpy as np
import random
import cv2
import tensorflow as tf
import pandas as pd

from tqdm import tqdm
from PIL import Image, ImageOps
from skimage import transform
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt





class FaceDetector:

    def __init__(self, model_file_path='', resize=(128,128), original_img_height=640, original_img_width=480, data_dir='data', imgs_dir='augmented_imgs', labels_dir='augmented_labels', estimator=None):

        self.model_file_path = model_file_path
        self.resize = resize
        self.original_img_height = original_img_height
        self.original_img_width = original_img_width
        self.data_dir = data_dir
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.estimator = estimator

    
    def transform(self, X, y=[], is_files=False, gryscl=True):

        """
        Preprocesses a collection of images

        Arguments:

            X: numpy array of shape (num_samples, width, height, num_channels)

            y: a numpy array of labels with shape (num_samples, class). Class 0 means no face, Class 1 means face

            is_files: whether X is an array of pixels or files

            gryscl: if true, reduces num_channels to 1
        
        Returns:

            preprocessed X with shape of (num_samples, width, height, 1 if grayscl else 3)

        """
        X_preproc = []

        if len(X) != 0:

            if is_files:

                
                for file_path in tqdm(X, desc="Transforming Data"):

                    dimensions = (self.original_img_width, self.original_img_height)
                    ultimate_path = os.path.join(self.data_dir, self.imgs_dir, file_path)
                    
                    if os.path.exists(ultimate_path):

                        img = Image.open(ultimate_path)

                        if self.resize:
                            img = img.resize(self.resize)
                            width, height = img.size
                            dimensions = (width, height)
                        
                        if gryscl:
                            img = ImageOps.grayscale(img)
                            img_data = np.array(img.getdata())
                            img_data = np.reshape(img_data, (dimensions[0], dimensions[1], 1))

                        else:
                            img_data = np.array(img.getdata())
                            img_data = np.reshape(img_data, (dimensions[0], dimensions[1], 3))
                        
                        img_data = img_data/255
                        X_preproc.append(img_data)
                    else:

                        print('Path to img is broken: ',ultimate_path)
            
            else:

                dimensions = (self.original_img_height, self.original_img_width)
                if self.resize:
                    dimensions = self.resize

                for img in tqdm(X, desc="Transforming Data"):

                    img = transform.resize(img, dimensions, mode='reflect', anti_aliasing=True) * 255
                    img = img.astype(np.uint8)

                    if gryscl:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = np.reshape(img, (1, dimensions[0], dimensions[1], 1))
                    else:
                        img = np.reshape(img, (1, dimensions[0], dimensions[1], 3))
                    
                    img = img/255
                    X_preproc.append(img)

            if len(y) != 0:

                zipped = list(zip(X_preproc, y))
                random.shuffle(zipped)
                X_preproc, y = zip(*zipped)

                X_preproc = np.array(X_preproc)
                y = np.array(y)
                return X_preproc, y
            
            else:

                random.shuffle(X_preproc)
                X_preproc = np.array(X_preproc)
                return X_preproc
        
        else:
            print("Empty X. Operation Canceled.")
    
    def fit(self, X, y, custom_NN=None, n_epochs=200, batch_size=64):

        """
        Fits an estimator to X and its labels. Outputs loss and accuracy plots at end of training

        Arguments:

            X: a numpy array of images with shape (num_samples, width, height, num_channels)
            
            y: a numpy array of labels with shape (num_samples, class). Class 0 means no face, Class 1 means face

            custom_NN: keras-built neural network imported from user

            n_epochs: number of epochs 

            batch_size: number of data samples per epoch

        Returns:

            final estimator: fully-trained model 
        """

        if X.size != 0 and y.size != 0:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
            X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, train_size=0.8, test_size=0.2, random_state=1)

            print('====Data Shapes=====')
            print('X_train shape: ',X_train.shape)
            print("X_val shape: ", X_val.shape)
            print(f"X_val class distribution: no_face {len(np.where(y_val == 0)[0])}, face {len(np.where(y_val == 1)[0])}")
            print("X_test shape: ",X_test.shape)
            print('====================')
            input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
            
            if custom_NN:
                model = custom_NN
            else:
                base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
                base_model.trainable = False

                flatten = base_model.output
                flatten = Flatten()(flatten)

                bboxHead = Dense(128, activation='relu')(flatten)
                bboxHead = Dense(64, activation='relu')(bboxHead)
                bboxHead = Dense(32, activation='relu')(bboxHead)
                prediction = Dense(len(y_train[0]), activation='sigmoid')(bboxHead)

                model = tf.keras.Model(inputs=base_model.input, outputs=prediction)

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
            model.summary()


            callbacks = EarlyStopping(monitor='val_loss', min_delta=0.001, restore_best_weights=True, patience=10)

            history = model.fit(X_train, y_train, batch_size=batch_size, callbacks=[callbacks], validation_data=(X_val, y_val), epochs=n_epochs)

            history_df = pd.DataFrame(history.history)
            plt.figure(figsize=(10, 6)) 

            loss_plot = history_df.plot(y=['loss', 'val_loss'], title="Cross-entropy Loss", style=['o-', 's-'])
            loss_plot.set_xlabel("Epoch")
            loss_plot.set_ylabel("Loss")  
            loss_plot.legend()

            
            plt.figure(figsize=(10, 6))  

            accuracy_plot = history_df.plot(y=['binary_accuracy', 'val_binary_accuracy'], title="Binary Accuracy", style=['o-', 's-'])
            accuracy_plot.set_xlabel("Epoch") 
            accuracy_plot.set_ylabel("Accuracy")  
            accuracy_plot.legend()  

            plt.show()

            _, acc = model.evaluate(X_test, y_test)
            print("Model accuracy on test set: ",acc*100)

            model.save(self.model_file_path)

            self.estimator = model

            return model
        else:
            print('Empty X and/or y. Operation Cancelled.')
    
    def predict(self, pixels):

        """
        Predicts whether face is in image or not

        Arguments:

            pixels: numpy array of image data
        
        Returns:

            class and raw prediction

        """

        if self.estimator:
            
            model = self.estimator

            raw_prediction = model.predict(pixels)

            predicted_class = np.round(raw_prediction, 0)

            return predicted_class, raw_prediction
        
        else:
            print("Current FaceTracker does not have estimator. Please provide estimator or fit a model.")


class BBoxPredictor:

    def __init__(self, model_file_path='', resize=(128,128), original_img_height=640, original_img_width=480, data_dir='data', imgs_dir='augmented_imgs', labels_dir='augmented_labels', estimator=None):

        self.model_file_path = model_file_path
        self.resize = resize
        self.original_img_height = original_img_height
        self.original_img_width = original_img_width
        self.data_dir = data_dir
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.estimator = estimator
    
    def transform(self, X, y=[], is_files=False, gryscl=True):

        """
        Preprocesses a collection of images

        Arguments:

            X: numpy array of shape (num_samples, width, height, num_channels)

            y: a numpy array of labels with shape (num_samples, class). Class 0 means no face, Class 1 means face

            is_files: whether X is an array of pixels or files

            gryscl: if true, reduces num_channels to 1
        
        Returns:

            preprocessed X with shape of (num_samples, width, height, 1 if grayscl else 3)

        """
        X_preproc = []

        if X.size != 0:

            if is_files:

                

                
                for file_path in tqdm(X, desc="Transforming Data for BBoxPredictor"):

                    dimensions = (self.original_img_width, self.original_img_height)
                    ultimate_path = os.path.join(self.data_dir, self.imgs_dir, file_path)
                    
                    if os.path.exists(ultimate_path):

                        img = Image.open(ultimate_path)

                        if self.resize:
                            img = img.resize(self.resize)
                            width, height = img.size
                            dimensions = (width, height)
                        
                        if gryscl:
                            img = ImageOps.grayscale(img)
                            img_data = np.array(img.getdata())
                            img_data = np.reshape(img_data, (dimensions[0], dimensions[1], 1))

                        else:
                            img_data = np.array(img.getdata())
                            img_data = np.reshape(img_data, (dimensions[0], dimensions[1], 3))
                        
                        img_data = img_data/255
                        X_preproc.append(img_data)
                    else:

                        print('Path to img is broken: ',ultimate_path)
                        continue
            
            else:

                dimensions = (self.original_img_height, self.original_img_width)
                if self.resize:
                    dimensions = self.resize

                for img in tqdm(X, desc="Transforming Data"):

                    img = transform.resize(img, dimensions, mode='reflect', anti_aliasing=True) * 255
                    img = img.astype(np.uint8)

                    if gryscl:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = np.reshape(img, (1, dimensions[0], dimensions[1], 1))
                    else:
                        img = np.reshape(img, (1, dimensions[0], dimensions[1], 3))

                    img = img/255
                    X_preproc.append(img)

            if y.size != 0:

                zipped = list(zip(X_preproc, y))
                random.shuffle(zipped)
                X_preproc, y = zip(*zipped)

                X_preproc = np.array(X_preproc)
                y = np.array(y)
                return X_preproc, y
            
            else:

                random.shuffle(X_preproc)
                X_preproc = np.array(X_preproc)
                return X_preproc
        
        else:
            print("Empty X. Operation Canceled.")
    
    def fit(self, X, y, custom_NN=None, n_epochs=200, batch_size=128, min_delta=0.01, patience=10):

        """
        Fits an estimator to X and its labels. Outputs loss and accuracy plots at end of training

        Arguments:

            X: a numpy array of images with shape (num_samples, width, height, num_channels)
            
            y: a numpy array of rectangle coordinates with format [xmin, ymin, xmax, ymax]

            custom_NN: keras-built neural network imported from user

            n_epochs: number of epochs 

            batch_size: number of data samples per epoch

        Returns:

            final estimator: fully-trained model 
        """

        if X.size != 0 and y.size != 0:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
            X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, train_size=0.8, test_size=0.2, random_state=1)

            print('====Data Shapes=====')
            print('X_train shape: ',X_train.shape)
            print("X_val shape: ", X_val.shape)
            print(f"X_val class distribution: no_face {len(np.where(y_val == 0)[0])}, face {len(np.where(y_val == 1)[0])}")
            print("X_test shape: ",X_test.shape)
            print('====================')
            input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
            
            if custom_NN:
                model = custom_NN
            else:
                base_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
                base_model.trainable = False

                flatten = base_model.output
                flatten = Flatten()(flatten)

                bboxHead = Dense(128, activation='relu')(flatten)
                bboxHead = Dense(64, activation='relu')(bboxHead)
                bboxHead = Dense(32, activation='relu')(bboxHead)
                prediction = Dense(len(y_train[0]), activation='sigmoid')(bboxHead)

                model = tf.keras.Model(inputs=base_model.input, outputs=prediction)

            model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
            model.summary()


            callbacks = EarlyStopping(monitor='val_loss', min_delta=min_delta, restore_best_weights=True, patience=patience)

            history = model.fit(X_train, y_train, batch_size=batch_size, callbacks=[callbacks], validation_data=(X_val, y_val), epochs=n_epochs)

            history_df = pd.DataFrame(history.history)
            plt.figure(figsize=(10, 6)) 

            loss_plot = history_df.plot(y=['loss', 'val_loss'], title="Mean Absolute Error", style=['o-', 's-'])
            loss_plot.set_xlabel("Epoch")
            loss_plot.set_ylabel("Loss")  
            loss_plot.legend()

            
            plt.figure(figsize=(10, 6))  

            accuracy_plot = history_df.plot(y=['accuracy', 'val_accuracy'], title="Accuracy", style=['o-', 's-'])
            accuracy_plot.set_xlabel("Epoch") 
            accuracy_plot.set_ylabel("Accuracy")  
            accuracy_plot.legend()  

            plt.show()

            _, acc = model.evaluate(X_test, y_test)
            print("Model accuracy on test set: ",acc*100)

            model.save(self.model_file_path)

            self.estimator = model

            return model
        else:
            print('Empty X and/or y. Operation Cancelled.')
    
    def predict(self, pixels):

        """
        Predicts the coordinates of a rectangle that would cover the face in the image

        Arguments:

            pixels: numpy array of image data
        
        Returns:

            rectangle coordinates in format [xmin, ymin, xmax, ymax]

        """

        if self.estimator:
            
            model = self.estimator

            raw_prediction = model.predict(pixels)

            return raw_prediction
        
        else:
            print("Current FaceTracker does not have estimator. Please provide estimator or fit a model.")


if __name__ == "__main__":
    print("File is meant for importing, not running.")



