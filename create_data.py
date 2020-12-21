import numpy as np
from numpy import save
import os
from PIL import Image

data_set = []

images = []
targets = []

for i in range(10):
    for root, dirs, files in os.walk('Dataset/' + str(i)):
        for file in files:

            image = Image.open('./Dataset/' + str(i) + '/' + file)
            data = np.asarray(image)

            if(data.shape == (100, 100, 3)):
                # images.append(data)
                # targets.append(i)

                data_set.append((data, i))

data_set = np.asarray(data_set, dtype=object)

np.random.shuffle(data_set)

for x in data_set:
    images.append(x[0])
    targets.append(x[1])

images = np.asarray(images, dtype='float64')
targets = np.asarray(targets, dtype='float64')

# print(images.shape)
# print(targets.shape)

save('data_set_images.npy', images)
save('data_set_targets.npy', targets)