# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
EPOCHS = 40
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)
data = []
labels = []
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(IMAGE_DIMS[1],IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
#print("labels")
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=0)
print("[INFO] serializing network...")
model.save("Teeth")
print("[INFO] serializing label binarizer...")
f = open("mlb.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
print("Training done")
