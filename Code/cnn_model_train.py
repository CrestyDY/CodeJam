import numpy as np
import pickle
import cv2, os
from glob import glob
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	# Try to find any image file in the gestures directory structure
	image_files = glob('gestures/*/*')
	if not image_files:
		# If no images found, return default size (you may need to adjust these values)
		print("Warning: No gesture images found. Using default size (50, 50)")
		return 50, 50
	
	# Try to read the first available image
	for img_path in image_files:
		img = cv2.imread(img_path, 0)
		if img is not None:
			return img.shape
	
	# If no readable images found, return default size
	print("Warning: No readable gesture images found. Using default size (50, 50)")
	return 50, 50

def get_num_of_classes():
	# Count only directories in gestures folder
	gesture_dirs = [d for d in os.listdir('gestures') if os.path.isdir(os.path.join('gestures', d))]
	return len(gesture_dirs)

image_x, image_y = get_image_size()

def cnn_model(num_of_classes):
	model = Sequential()
	model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
	model.add(Conv2D(32, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
	model.add(Conv2D(64, (5,5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_of_classes, activation='softmax'))
	sgd = optimizers.SGD(learning_rate=1e-2)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	filepath="cnn_model_keras2.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]
	#from keras.utils import plot_model
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model, callbacks_list

def train():
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.int32)

	with open("val_images", "rb") as f:
		val_images = np.array(pickle.load(f))
	with open("val_labels", "rb") as f:
		val_labels = np.array(pickle.load(f), dtype=np.int32)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
	
	# Determine number of classes from the actual labels
	# Get unique labels from both train and val sets to ensure we have all classes
	all_labels = np.concatenate([train_labels, val_labels])
	unique_labels = np.unique(all_labels)
	
	# If labels don't start from 0, we need to remap them
	label_min = np.min(all_labels)
	if label_min != 0:
		print(f"Warning: Labels don't start from 0 (min={label_min}). Remapping to start from 0.")
		# Create a mapping from old labels to new labels (0-indexed)
		label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
		train_labels = np.array([label_map[label] for label in train_labels], dtype=np.int32)
		val_labels = np.array([label_map[label] for label in val_labels], dtype=np.int32)
		num_of_classes = len(unique_labels)
		print(f"Remapped labels. New label range: 0 to {num_of_classes-1}")
	else:
		num_of_classes = int(np.max(all_labels) + 1)
	
	print(f"Number of classes: {num_of_classes}")
	print(f"Unique original labels: {sorted(unique_labels)}")
	
	train_labels = to_categorical(train_labels, num_classes=num_of_classes)
	val_labels = to_categorical(val_labels, num_classes=num_of_classes)

	print(f"Train labels shape: {train_labels.shape}")
	print(f"Val labels shape: {val_labels.shape}")

	model, callbacks_list = cnn_model(num_of_classes)
	model.summary()
	model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=500, callbacks=callbacks_list)
	scores = model.evaluate(val_images, val_labels, verbose=0)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	#model.save('cnn_model_keras2.h5')

train()
K.clear_session();
