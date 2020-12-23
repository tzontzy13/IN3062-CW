import numpy as np
from numpy import save
import os
from PIL import Image

# lists for images and targets
data_set = []

images = []
targets = []

# range 10 because we have 10 folders of images, from 0 to 9
for i in range(10):
    # each folder has about 205 images, we add 170 to the train set and keep the rest for testing
    for root, dirs, files in os.walk('Dataset/' + str(i)):
        for file in files:

            image = Image.open('./Dataset/' + str(i) + '/' + file)
            data = np.asarray(image)

            if(data.shape == (100, 100, 3)):
                # append image and index
                data_set.append((data, i))

# transform to numpy array and shuffle
data_set = np.asarray(data_set, dtype=object)

np.random.shuffle(data_set)

# split data into images and targets
for x in data_set:
    images.append(x[0])
    targets.append(x[1])

images = np.asarray(images, dtype='float64')
targets = np.asarray(targets, dtype='float64')

# save data in .npy files
save('data_set_images.npy', images)
save('data_set_targets.npy', targets)