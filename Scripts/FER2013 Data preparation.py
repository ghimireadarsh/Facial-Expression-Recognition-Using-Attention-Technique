import sys
import os
import numpy as np
from keras.preprocessing.image import img_to_array, array_to_img
"""
    0 = Angry, 
    1 = Disgust, 
    2 = Fear, 
    3 = Happy,
    4 = Sad,
    5 = Surprise,
    6 = Neutral
"""

label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# CSV data loading
data = np.genfromtxt('../Data/fer2013/fer2013.csv', delimiter=',',dtype=None, encoding='utf8')

# Labels of images
labels = data[1:,0].astype(np.int8)
# Images pixel values
image_buffer = data[1:,1]

# Converting all the strings form of pixels into array
images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in image_buffer])

# Training, PrivateTesting, PublicTesting
usage = data[1:, 2]
dataset = zip(labels, images, usage)

output_path = '../Data/fer2013/'
for i, d in enumerate(dataset):
    usage_path = os.path.join(output_path, d[-1])
    label_path = os.path.join(usage_path, label_names[d[0]])
    img = array_to_img(d[1].reshape((48, 48, 1)))
    img_name = '%d.jpg' % i
    img_path = os.path.join(label_path, img_name)
    if not os.path.exists(usage_path):
        os.system('mkdir {}'.format(usage_path))
    if not os.path.exists(label_path):
        os.system('mkdir {}'.format(label_path))
    img.save(img_path)
    print('Write {}'.format(img_path))





