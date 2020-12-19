import numpy as np
from numpy import save
import os
from PIL import Image

data_set_train = []

images_train = []
targets_train = []

data_set_test = []

images_test = []
targets_test = []

for i in range(10):
    for root, dirs, files in os.walk('Dataset/' + str(i)):
        for file in files[0:170]:

            image = Image.open('./Dataset/' + str(i) + '/' + file)
            data = np.asarray(image)

            if(data.shape == (100, 100, 3)):
                # images.append(data)
                # targets.append(i)

                data_set_train.append((data, i))

        for file in files[171:]:

            image = Image.open('./Dataset/' + str(i) + '/' + file)
            data = np.asarray(image)

            if(data.shape == (100, 100, 3)):
                # images.append(data)
                # targets.append(i)

                data_set_test.append((data, i))

data_set_train = np.asarray(data_set_train, dtype=object)
np.random.shuffle(data_set_train)

data_set_test = np.asarray(data_set_test, dtype=object)
np.random.shuffle(data_set_test)

for x in data_set_train:
    images_train.append(x[0])
    targets_train.append(x[1])

for x in data_set_test:
    images_test.append(x[0])
    targets_test.append(x[1])

images_train = np.asarray(images_train, dtype='float64')
targets_train = np.asarray(targets_train, dtype='float64')

images_test = np.asarray(images_test, dtype='float64')
targets_test = np.asarray(targets_test, dtype='float64')
save('train_images.npy', images_train)
save('train_targets.npy', targets_train)
save('test_images.npy', images_test)
save('test_targets.npy', targets_test)
