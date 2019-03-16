"""Cactus identification challenge"""
from os import listdir
from os.path import isfile, join
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cv2
import pandas as pd


# Hide those silly TF warnings
from os import environ
from tensorflow.logging import set_verbosity, ERROR
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_verbosity(ERROR)


# Set up training data images and labels
files = [f for f in listdir('data/train') if isfile(join('data/train', f))]
X = []

for f in files:
  X.append(cv2.imread(join('data/train', f)))

X = np.array(X, dtype=np.float32) / 255.0
y = pd.read_csv('data/train.csv').to_numpy()[:,1]
y = np.array(y, dtype=np.int32)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
scores = []

# X_train = np.array(X_all[:-500], dtype=np.float32) / 255.0
# y_train = y_all[:-500]

# X_test = np.array(X_all[-500:])
# y_test = y_all[-500:]

for train, test in kfold.split(X, y):
  # Set up the model
  model = Sequential()

  model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32,32,3)))
  model.add(MaxPooling2D(pool_size=2, strides=2))
  model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=2, strides=2))
  model.add(Flatten())
  model.add(Dense(1000, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  model.fit(X[train], y[train], epochs=3)

  # Evaluate the model (on the training data for now)
  _, accuracy = model.evaluate(X[test], y[test], verbose=0)
  print('Accuracy: %f' % (accuracy*100))
  scores.append(accuracy * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))