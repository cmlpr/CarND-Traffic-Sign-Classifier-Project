
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle
import csv
import pprint
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib.gridspec as gridspec

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./raw_data/train.p"
testing_file = "./raw_data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
x_train, y_train = train['features'], train['labels']
x_test, y_test = test['features'], test['labels']

# CSV file for sign names
names_file = "./raw_data/signnames.csv"

# Load class names into dictionary
with open(names_file, mode='r') as f:
    reader = csv.reader(f)
    next(reader)  # skip the header
    signnames = {rows[0]: rows[1] for rows in reader}

# Shuffle data used in model training
x_train, y_train = shuffle(x_train, y_train)

# Make sure the x and y data have same length
assert (len(x_train) == len(y_train))
assert (len(x_test) == len(y_test))
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 2D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below.


```python
### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(x_train)

# TODO: Number of testing examples.
n_test = len(x_test)

# TODO: What's the shape of an traffic sign image?
image_shape = x_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Count the number of examples in each class and print them in detail with class description
unique, counts = np.unique(y_train, return_counts=True)
class_info = {}
for unique, counts in zip(unique, counts):
    class_info[str(unique)] = {'description': signnames[str(unique)], 'count': counts}

pp = pprint.PrettyPrinter(indent=4, width=100)
print("Example Count in Each Class\n")
pp.pprint(class_info)
```

    Number of training examples = 39209
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43
    Example Count in Each Class
    
    {   '0': {'count': 210, 'description': 'Speed limit (20km/h)'},
        '1': {'count': 2220, 'description': 'Speed limit (30km/h)'},
        '10': {'count': 2010, 'description': 'No passing for vehicles over 3.5 metric tons'},
        '11': {'count': 1320, 'description': 'Right-of-way at the next intersection'},
        '12': {'count': 2100, 'description': 'Priority road'},
        '13': {'count': 2160, 'description': 'Yield'},
        '14': {'count': 780, 'description': 'Stop'},
        '15': {'count': 630, 'description': 'No vehicles'},
        '16': {'count': 420, 'description': 'Vehicles over 3.5 metric tons prohibited'},
        '17': {'count': 1110, 'description': 'No entry'},
        '18': {'count': 1200, 'description': 'General caution'},
        '19': {'count': 210, 'description': 'Dangerous curve to the left'},
        '2': {'count': 2250, 'description': 'Speed limit (50km/h)'},
        '20': {'count': 360, 'description': 'Dangerous curve to the right'},
        '21': {'count': 330, 'description': 'Double curve'},
        '22': {'count': 390, 'description': 'Bumpy road'},
        '23': {'count': 510, 'description': 'Slippery road'},
        '24': {'count': 270, 'description': 'Road narrows on the right'},
        '25': {'count': 1500, 'description': 'Road work'},
        '26': {'count': 600, 'description': 'Traffic signals'},
        '27': {'count': 240, 'description': 'Pedestrians'},
        '28': {'count': 540, 'description': 'Children crossing'},
        '29': {'count': 270, 'description': 'Bicycles crossing'},
        '3': {'count': 1410, 'description': 'Speed limit (60km/h)'},
        '30': {'count': 450, 'description': 'Beware of ice/snow'},
        '31': {'count': 780, 'description': 'Wild animals crossing'},
        '32': {'count': 240, 'description': 'End of all speed and passing limits'},
        '33': {'count': 689, 'description': 'Turn right ahead'},
        '34': {'count': 420, 'description': 'Turn left ahead'},
        '35': {'count': 1200, 'description': 'Ahead only'},
        '36': {'count': 390, 'description': 'Go straight or right'},
        '37': {'count': 210, 'description': 'Go straight or left'},
        '38': {'count': 2070, 'description': 'Keep right'},
        '39': {'count': 300, 'description': 'Keep left'},
        '4': {'count': 1980, 'description': 'Speed limit (70km/h)'},
        '40': {'count': 360, 'description': 'Roundabout mandatory'},
        '41': {'count': 240, 'description': 'End of no passing'},
        '42': {'count': 240, 'description': 'End of no passing by vehicles over 3.5 metric tons'},
        '5': {'count': 1860, 'description': 'Speed limit (80km/h)'},
        '6': {'count': 420, 'description': 'End of speed limit (80km/h)'},
        '7': {'count': 1440, 'description': 'Speed limit (100km/h)'},
        '8': {'count': 1410, 'description': 'Speed limit (120km/h)'},
        '9': {'count': 1470, 'description': 'No passing'}}


Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.


```python
### Data exploration visualization goes here.
# Visualizations will be shown in the notebook.
%matplotlib inline

#
# Plot the first image
#
plt.figure(1)
# plt.imshow(x_train[0].squeeze(), cmap='gray')
plt.imshow(x_train[0])
plt.title(class_info[str(y_train[0])]['description'])
plt.savefig('./images/first_image')
print('\nCompleted plotting the first image')
```


```python
#
# Plot random 16 images
#
plt.figure(2)
grid = np.random.randint(n_train, size=(4, 4))
fig, axes = plt.subplots(4, 4, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.2, wspace=0.05)
fig.suptitle('Random 16 images', fontsize=20)
for ax, i in zip(axes.flat, list(grid.reshape(16, 1))):
    # ax.imshow(x_train[int(i)].squeeze(), cmap='gray')
    ax.imshow(x_train[int(i)])
    title = str(i) + " - " + class_info[str(y_train[int(i)])]['description']
    ax.set_title(title, fontsize=8)
plt.show()
plt.savefig('./images/16_random_images')
plt.close()
print('\nCompleted plotting the random 16 images')
```


    <matplotlib.figure.Figure at 0x7f7794b7f8d0>



![png](output_7_1.png)


    
    Completed plotting the random 16 images



```python
#
# Bar plot showing the count of each class in the training set
#
unique, counts = np.unique(y_train, return_counts=True)
plt.figure(3)
plt.bar(unique, counts, 0.5, color='b')
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Frequency of Each Class')
plt.show()
plt.savefig('./images/class_freq_plot')
print('\nCompleted plotting class frequency bar plot')
```


![png](output_8_0.png)


    
    Completed plotting class frequency bar plot



    <matplotlib.figure.Figure at 0x7f7790703780>


As can be seen in the  histogram plot above, there is class imbalance in the training data. Some classes have over 1500 examples and some have below 500 examples. This can hurt the testing accuracy of the model. 

----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

There are various aspects to consider when thinking about this problem:

- Neural network architecture
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

**NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

### Implementation

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.


```python
def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')
    :param img: image to be converted to grayscale
    :return: image in grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```


```python
def hist_equalize(img):
    """
    Improve the contrast of image
    Helps distribute the range of color in the image
    Read more at
    http://docs.opencv.org/trunk/d5/daf/tutorial_py_histogram_equalization.html
    :param img:
    :return:
    """
    return cv2.equalizeHist(img)
```


```python
def normalize_scale(img):
    """
    Normalize images by subtracting mean and dividing by the range so that pixel values are between -0.5 and 0.5
    :param img:
    :return:
    """
    # normalized_image = np.divide(img - 125.0, 255.0)
    normalized_image = (img - 125.0) / 255.0
    return normalized_image
```


```python
def pre_processing(img_list):
    """
    Call the grayscale, histogram equalization and normalization functions in the order and return images
    with single channel
    :param img_list:
    :return:
    """
    count = len(img_list)
    shape = img_list[0].shape
    processed = []

    for i in range(count):
        img = normalize_scale(hist_equalize(grayscale(img_list[i])))
        processed.append(img)
    
    print("\nPreprocessing of images complete..\n")
    
    return np.reshape(np.array(processed), [count, shape[0], shape[1], 1])
```


```python
def image_augmentation(img):

    # References
    # Geometric Transformations
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    # Morphological Transformation
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    # Smoothing Images
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

    # Perform smoothing or morphological transformation
    picker = np.random.randint(low=0, high=10)

    if picker == 1:
        # Erosion
        # Erodes away the boundaries of foreground objects - white region decreases in the image
        erosion_kernel = np.ones((2, 2), np.uint8)
        img_mod = cv2.erode(img, erosion_kernel, iterations=1)
    elif picker == 2:
        # Dilation
        # Opposite of erosion
        dilation_kernel = np.ones((2, 2), np.uint8)
        img_mod = cv2.dilate(img, dilation_kernel, iterations=1)
    elif picker == 3:
        # Opening
        # Erosion followed by dilation - removes noise
        opening_kernel = np.ones((3, 3), np.uint8)
        img_mod = cv2.morphologyEx(img, cv2.MORPH_OPEN, opening_kernel)
    elif picker == 4:
        # Closing
        # Reverse of opening - dilation followed by erosion
        closing_kernel = np.ones((3, 3), np.uint8)
        img_mod = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closing_kernel)
    # elif picker == 5:
    #     # Morphological Gradient
    #     # Difference between the dilation and erosion of an image
    #     gradient_kernel = np.ones((3, 3), np.uint8)
    #     img_mod = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, gradient_kernel)
    # elif picker == 6:
    #     # Top hat
    #     # Difference between the input image and the opening of the image
    #     tophat_kernel = np.ones((7, 7), np.uint8)
    #     img_mod = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, tophat_kernel)
    elif picker == 7:
        # Blur
        img_mod = cv2.blur(img, (2, 2))
    elif picker == 8:
        # Gaussian blur
        img_mod = cv2.GaussianBlur(img, (3, 3), 0)
    elif picker == 9:
        # Median Blur
        img_mod = cv2.medianBlur(img, 3)
    else:
        img_mod = img

    # Rotation:
    max_rotation_angle = 30.0  # degrees
    max_center_translation = 5.0  # pixels
    angle = np.random.uniform(low=-max_rotation_angle, high=max_rotation_angle)
    center = tuple(np.array(img_mod.shape[0:2]) / 2.0)
    center = (center[0] + np.random.uniform(low=-max_center_translation, high=max_center_translation),
              center[1] + np.random.uniform(low=-max_center_translation, high=max_center_translation))

    # Translation:
    max_translation = 5.0  # pixels
    x_translation = np.random.uniform(low=-max_translation, high=max_translation)
    y_translation = np.random.uniform(low=-max_translation, high=max_translation)
    translation_matrix = np.float32([[1, 0, x_translation],
                                     [0, 1, y_translation]])
    # Affine transformation:
    # pts1 = np.float32([[5, 5],
    #                    [20, 5],
    #                    [5, 20]])
    # pts2 = np.float32([[1, 10],
    #                    [20, 5],
    #                    [10, 25]])
    # affine_transform_matrix = cv2.getAffineTransform(pts1, pts2)

    # Perspective Transformation
    # pts1 = np.float32([[3, 4], [29, 3], [4, 28], [27, 27]])
    # pts2 = np.float32([[0, 0], [22, 0], [0, 22], [22, 22]])
    # perspective_transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_mod = cv2.warpAffine(img_mod, rot_mat, img_mod.shape[0:2], flags=cv2.INTER_LINEAR)
    img_mod = cv2.warpAffine(img_mod, translation_matrix, img_mod.shape[0:2])
    # result = cv2.warpAffine(img_mod, affine_transform_matrix, img_mod.shape[0:2])
    # result = cv2.warpAffine(img_mod, perspective_transform_matrix, (22, 22))

    # Brightness
    brightness_multiplier = np.random.uniform(low=-0.25, high=0.25)
    hsv = cv2.cvtColor(img_mod, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + brightness_multiplier)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(result)
    # plt.show()

    return result
```


```python
def augment_training_images(x, y, params):
    # Augment training set image set
    if params['augment']:

        target_class_size = params['augmented_class_size']

        unq, unq_inv, unq_cnt = np.unique(y, return_inverse=True, return_counts=True)
        class_index = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))

        new_training_x = []
        new_training_y = []
        for cls in unq:
            if unq_cnt[cls] < target_class_size:
                new_img_count = target_class_size - unq_cnt[cls]
                for i in range(new_img_count):
                    pck = np.random.choice(class_index[cls])
                    new_training_x.append(image_augmentation(x[pck]))
                    new_training_y.append(y[pck])
        x = np.vstack((x, np.asarray(new_training_x)))
        y = np.hstack((y, np.array(new_training_y)))

    print("\n Extra training images are generated... \n")

    return x, y
```


```python
# Pre - processing parameters
pre_process_param = {
    'pre-process': True,
    'mode': 1,
    'augment': False,
    'augmented_class_size': 3500
}
x_train, y_train = augment_training_images(x_train, y_train, pre_process_param)
x_train = pre_processing(x_train)
x_test = pre_processing(x_test)
```

    
     Extra training images are generated... 
    
    
    Preprocessing of images complete..
    
    
    Preprocessing of images complete..
    



```python
#
# Bar plot showing the count of each class in the training set after augmentation
#
unique, counts = np.unique(y_train, return_counts=True)
plt.figure(3)
plt.bar(unique, counts, 0.5, color='b')
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Frequency of Each Class after Augmentation')
plt.show()
plt.savefig('./images/class_freq_plot_after_augment')
print('\nCompleted plotting class frequency bar plot')
```


![png](output_19_0.png)


    
    Completed plotting class frequency bar plot



    <matplotlib.figure.Figure at 0x7f7790893828>


### Question 1 

_Describe how you preprocessed the data. Why did you choose that technique?_

**Answer:** 

Generating new images can be helpful to  eliminate the class imbalance problem.  New images can be generated by jittering the current images and adding them to the trainign set. Some of the techniques that are selected here: 

geometric transformations: translation and rotation
morphological transformations: erosion, dilation, opening, closing
smoothing: blur, gaussian blur and median blur
brightness adjustment

Each class size is increased to 3500 by randomly selecting images from the original training set and applying the  jittering techniques randomly. 

In the second step, each image is converted to grayscale. This is followed by histogram normalization. Images can use a narrow band of pixel values in the total range of 0-255.  Histogram  normalization helps distribute the intensities so that the whole range is utilized in the images. The last step for preprocessing was normalization. Each image is brough to -0.5 to 0.5 range by subtracting mean and  dividing by 255(range/max). This helps prevent getting large numbers after multiplications and create equal range for each training examples so that there is no bias among data.


```python
### Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

x_train, y_train = shuffle(x_train, y_train)

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=832289)
    
n_train = len(x_train)
n_valid = len(x_valid)
n_test = len(x_test)
    
print("Training Set:   {} samples".format(n_train))
print("Validation Set: {} samples".format(n_valid))
print("Test Set:       {} samples".format(n_test))
```

    Training Set:   31367 samples
    Validation Set: 7842 samples
    Test Set:       12630 samples


### Question 2

_Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_

**Answer:**

Shuffled and split the data to 80% training and 20% validation sets. 


```python
### Define your architecture here.
```


```python
# Implement LeNet - 5
def lenet_model1(data_x, params, channel_count, keep_prob):

    # The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels.

    # Architecture
    # Layer 1: Convolutional. The output shape is 28x28x32.
    # Activation function
    # Pooling. The output shape is 14x14x32.
    # Layer 2: Convolutional. The output shape should be 10x10x64.
    # Activation function
    # Pooling. The output shape is 5x5x64.
    # Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
    # The easiest way to do is by using tf.contrib.layers.flatten
    # Layer 3: Fully Connected. This has 532 outputs.
    # Activation function
    # Layer 4: Fully Connected. This has 532 outputs.
    # Activation function
    # Layer 5: Fully Connected (Logits). 43 class outputs.
    # Output
    # Return the result of the 2nd fully connected layer.

    # Hyperparameters
    mu = params['mean']
    sigma = params['std']
    chn = channel_count

    layer_depth = {
        'conv_1': 32,
        'conv_2': 64,
        'full_1': 1064,
        'full_2': 532,
        'out': params['class_count']
    }

    # Store layers weight & bias
    weights = {
        'conv_1': tf.Variable(tf.truncated_normal([5, 5, chn, layer_depth['conv_1']], mean=mu, stddev=sigma)),
        'conv_2': tf.Variable(tf.truncated_normal([5, 5, layer_depth['conv_1'], layer_depth['conv_2']],
                                                  mean=mu, stddev=sigma)),
        'full_1': tf.Variable(tf.truncated_normal([5 * 5 * layer_depth['conv_2'], layer_depth['full_1']], mean=mu, stddev=sigma)),
        'full_2': tf.Variable(tf.truncated_normal([layer_depth['full_1'], layer_depth['full_2']],
                                                  mean=mu, stddev=sigma)),
        'out':    tf.Variable(tf.truncated_normal([layer_depth['full_2'], layer_depth['out']],
                                                  mean=mu, stddev=sigma))
    }
    biases = {
        'conv_1': tf.Variable(tf.zeros(layer_depth['conv_1'])),
        'conv_2': tf.Variable(tf.zeros(layer_depth['conv_2'])),
        'full_1': tf.Variable(tf.zeros(layer_depth['full_1'])),
        'full_2': tf.Variable(tf.zeros(layer_depth['full_2'])),
        'out':    tf.Variable(tf.zeros(layer_depth['out']))
    }

    # Layer 1: Convolutional. Input = 32x32xchn. Output = 28x28xlayer_depth['conv_1'].
    conv1 = tf.nn.conv2d(data_x, weights['conv_1'], strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, biases['conv_1'])

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28xlayer_depth['conv_1']. Output = 14x14xlayer_depth['conv_1'].
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # conv1 = tf.nn.dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Output = 10x10xlayer_depth['conv_2'].
    conv2 = tf.nn.conv2d(conv1, weights['conv_2'], strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, biases['conv_2'])

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10xlayer_depth['conv_2']. Output = 5x5xlayer_depth['conv_2'].
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # conv2 = tf.nn.dropout(conv2, keep_prob)

    # Flatten. Input = 5x5xlayer_depth['conv_2']. Output = 1600.
    fc1 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 1600. Output = layer_depth['full_1'].
    fc1 = tf.add(tf.matmul(fc1, weights['full_1']), biases['full_1'])

    # Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = layer_depth['full_1']. Output = layer_depth['full_2'].
    fc2 = tf.add(tf.matmul(fc1, weights['full_2']), biases['full_2'])

    # Activation.
    fc2 = tf.nn.relu(fc2)
    # fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = layer_depth['full_2']. Output = class_count.
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    reg_term = params['l2_beta'] * (tf.nn.l2_loss(weights['conv_1']) + tf.nn.l2_loss(weights['conv_2']) +
                                    tf.nn.l2_loss(weights['full_1']) + tf.nn.l2_loss(weights['full_2']))

    return logits, reg_term
```

### Question 3

_What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._


**Answer:**

The implementation has a similar architecture to Lenet-5. The difference is the depth of convolution layers and the  fully connected layers. 

The input is 32x32xC image as input, where C is the number of color channels. In the final form single channel images are used so C = 1

Architecture
Layer 1: Convolutional: Filter = 5x5
                        Strides = 1x1
                        Padding = Valid
                        The output shape is 28x28x32

         Activation function is RELU

         Pooling: Kernel size = 2x2
                  Strides = 2x2
                  Padding = VALID 
                  The output shape is 14x14x32

Layer 2: Convolutional: Filter = 5x5
                        Strides = 1x1
                        Padding = Valid
                        The output shape is 10x10x64

         Activation function is RELU

         Pooling: Kernel size = 2x2
                  Strides = 2x2
                  Padding = VALID 
                  The output shape is 5x5x64

Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D using tf.contrib.layers.flatten. 

Layer 3: Fully Connected 
         532 outputs
         Dropout with 0.5
         
         Activation function is RELU
         
Layer 4: Fully Connected. 
         532 outputs
         
         Activation function is RELU
        
Layer 5: Fully Connected (Logits)
         43 class outputs

Output


```python
### Train your model here.

model_param_list = {
        'name': 'trial_1',
        'epoch': 30,
        'batch_size': 128,
        'mean': 0.,
        'std': 0.1,
        'class_count': len(class_info),
        'rate': 0.001,
        'l2_beta': 0.005,
        'dropout_prob': 0.7
}

model_name = model_param_list['name']
channel_count = x_train[0].shape[2]

# Placeholder for batch of input images
model_x = tf.placeholder(tf.float32, (None, 32, 32, channel_count))
# Placeholder for batch of output labels
model_y = tf.placeholder(tf.int32, None)

# One hot encode the training set - one vs all
one_hot_y = tf.one_hot(model_y, model_param_list['class_count'])

# Dropout only
keep_prob = tf.placeholder(tf.float32)

result_logits, reg_adder = lenet_model1(model_x, model_param_list, channel_count, keep_prob)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(result_logits, one_hot_y)

loss_operation = tf.reduce_mean(cross_entropy) + reg_adder
    
optimizer = tf.train.AdamOptimizer(learning_rate=model_param_list['rate'])

training_operation = optimizer.minimize(loss_operation)

# Model evaluation
correct_prediction = tf.equal(tf.argmax(result_logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Save
saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Run the training data through the pipeline to train the model
# Before each epoch, shuffle the training set
# After each epoch, measure the loss and accuracy on the validation set
# Save the model after training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)
    print("\nTraining Model: {}\n".format(model_name))
    for i in range(model_param_list['epoch']):
        train_x, train_y = shuffle(x_train, y_train)
        for offset in range(0, num_examples, model_param_list['batch_size']):
            end = offset + model_param_list['batch_size']
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={model_x: batch_x, model_y: batch_y,
                                                    keep_prob: model_param_list['dropout_prob']})

        num_valid_examples = len(x_valid)
        total_accuracy = 0.0
        total_loss = 0.0
        for offset2 in range(0, num_valid_examples, model_param_list['batch_size']):
            batch_valid_x, batch_valid_y = x_valid[offset2:offset2 + model_param_list['batch_size']], \
                                           y_valid[offset2:offset2 + model_param_list['batch_size']]
            accuracy, lss = sess.run([accuracy_operation, loss_operation],
                                feed_dict={model_x: batch_valid_x, model_y: batch_valid_y, keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_valid_x))
            total_loss += (lss * len(batch_valid_x))
        
        validation_accuracy = total_accuracy / num_valid_examples
        validation_loss = total_loss / num_valid_examples

        print("EPOCH {}: Validation Accuracy = {:.3f}, Validation Loss = {:.3f}".format(i + 1, 
                                                                                        validation_accuracy,
                                                                                        validation_loss))

    saver.save(sess, './models/' + model_name)
```

    
    Training Model: trial_1
    
    EPOCH 1: Validation Accuracy = 0.909, Validation Loss = 6.386
    EPOCH 2: Validation Accuracy = 0.949, Validation Loss = 1.869
    EPOCH 3: Validation Accuracy = 0.944, Validation Loss = 0.994
    EPOCH 4: Validation Accuracy = 0.947, Validation Loss = 0.729
    EPOCH 5: Validation Accuracy = 0.918, Validation Loss = 0.702
    EPOCH 6: Validation Accuracy = 0.938, Validation Loss = 0.598
    EPOCH 7: Validation Accuracy = 0.945, Validation Loss = 0.552
    EPOCH 8: Validation Accuracy = 0.962, Validation Loss = 0.491
    EPOCH 9: Validation Accuracy = 0.945, Validation Loss = 0.515
    EPOCH 10: Validation Accuracy = 0.961, Validation Loss = 0.471
    EPOCH 11: Validation Accuracy = 0.970, Validation Loss = 0.436
    EPOCH 12: Validation Accuracy = 0.971, Validation Loss = 0.419
    EPOCH 13: Validation Accuracy = 0.962, Validation Loss = 0.439
    EPOCH 14: Validation Accuracy = 0.971, Validation Loss = 0.408
    EPOCH 15: Validation Accuracy = 0.964, Validation Loss = 0.414
    EPOCH 16: Validation Accuracy = 0.971, Validation Loss = 0.389
    EPOCH 17: Validation Accuracy = 0.979, Validation Loss = 0.377
    EPOCH 18: Validation Accuracy = 0.963, Validation Loss = 0.408
    EPOCH 19: Validation Accuracy = 0.978, Validation Loss = 0.352
    EPOCH 20: Validation Accuracy = 0.970, Validation Loss = 0.379
    EPOCH 21: Validation Accuracy = 0.949, Validation Loss = 0.426
    EPOCH 22: Validation Accuracy = 0.982, Validation Loss = 0.332
    EPOCH 23: Validation Accuracy = 0.980, Validation Loss = 0.332
    EPOCH 24: Validation Accuracy = 0.981, Validation Loss = 0.325
    EPOCH 25: Validation Accuracy = 0.979, Validation Loss = 0.340
    EPOCH 26: Validation Accuracy = 0.976, Validation Loss = 0.331
    EPOCH 27: Validation Accuracy = 0.970, Validation Loss = 0.351
    EPOCH 28: Validation Accuracy = 0.984, Validation Loss = 0.299
    EPOCH 29: Validation Accuracy = 0.983, Validation Loss = 0.293
    EPOCH 30: Validation Accuracy = 0.980, Validation Loss = 0.314



```python
### Testing

print("\nTesting Model: {}\n".format(model_name))

load_file = './models/' + model_name

with tf.Session() as sess:
    saver.restore(sess, load_file)

    num_test_examples = len(x_test)
    total_accuracy = 0.0
    total_loss = 0.0
    
    for offset in range(0, num_test_examples, model_param_list['batch_size']):
        batch_test_x, batch_test_y = x_test[offset:offset + model_param_list['batch_size']], \
                                     y_test[offset:offset + model_param_list['batch_size']]
        accuracy, lss = sess.run([accuracy_operation, loss_operation],
                            feed_dict={model_x: batch_test_x, model_y: batch_test_y, keep_prob: 1.0})
        
        total_accuracy += (accuracy * len(batch_test_y))
        total_loss += (lss * len(batch_test_y))
    
    test_accuracy = total_accuracy / num_test_examples
    test_loss = total_loss / num_test_examples

    print("Test Accuracy = {:.3f}, Test Loss = {:.3f}".format(test_accuracy, test_loss))
```

    
    Testing Model: trial_1
    
    Test Accuracy = 0.930, Test Loss = 0.473


### Question 4

_How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_


**Answer:**

model_param_list = {
        'name': 'trial_1',
        'epoch': 30,
        'batch_size': 128,
        'mean': 0.,
        'std': 0.1,
        'class_count': len(class_info),
        'rate': 0.001,
        'l2_beta': 0.01,
        'dropout_prob': 0.5
}

Hyperparameters are listed in a dictionary called "model_param_list". 
Epoch size is 30 and batch size is 128
For parameter initialization a mean of 0 and a small std deviation of 0.1 are found to be working better.
Learning rate is chosen to be 0.001.
Regularization is applied in the loss operation  with a multiplier 0.01
Dropout is also applied at the end of the first fully connected layer with probability 0.5.
Both regularization and dropout will help with the overfitting. 

### Question 5


_What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem._

**Answer:**

LeNet is used as the starting model with larger the layer depths based on the article provided in this notebook. CNNs are appropriate in the image classification problems. The images were simple and using 2 convolutional layers should be enough. A 3rd layer was added in one of the trials but did not generate better training performance (for the selected hyperparameters). The process is mostly trial - error while considering memory and time constraints. Decisions were made based on the accuracy and loss results. 


---

## Step 3: Test a Model on New Images

Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Implementation

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.


```python
# First let's see the images for each class and their description

unq, unq_inv, unq_cnt = np.unique(y_train, return_inverse=True, return_counts=True)
class_index = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))

n_size = len(unq)
grid = np.zeros((n_size, 8), dtype=int)
for cls in range(n_size):
    grid[cls, :] = np.random.choice(class_index[cls], size=8, replace=False)
grid_list = [int(i) for i in list(grid.reshape(n_size*8, 1))]

plt_grid = gridspec.GridSpec(43, 8)
plt_grid.update(wspace=0.01, hspace=0.02)
plt.figure(figsize=(5,25))

cnt = 0
for i in range(43*8):
    cnt += 1
    ax1 = plt.subplot(plt_grid[i])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    plt.subplot(43,8,i+1)
    plt.imshow(x_train[grid_list[i]].squeeze(), cmap='gray')
    if cnt % 8 == 1:
        title = str(y_train[grid_list[i]]) + " - " + class_info[str(y_train[grid_list[i]])]['description']
        plt.title(title, fontsize = 8)
    plt.axis('off')

plt.show()
```


![png](output_37_0.png)



```python
### Load the images and plot them here.

x_new_test = []
for i in range(7):
    img_path = './web_img/test_img' + str(i+1) + '.jpg'
    test_img = cv2.imread(img_path)
    x_new_test.append(test_img)

x_new_test = np.asarray(x_new_test)

x_new_test = pre_processing(x_new_test)
```

    
    Preprocessing of images complete..
    


### Question 6

_Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook._



**Answer:**

New signs are obtaiend from  streets of Frankfurt using Google Maps. 4 out of 7 images are similar to the images in training set however 3 images are not currently in the training set. The model should be able to correctly predict 4 images at max. 


```python
### Run the predictions here.

load_file = './models/' + model_name

y_pred_prob = tf.nn.softmax(result_logits)
top5_pred_prob = tf.nn.top_k(y_pred_prob, 5)
y_pred_class = tf.argmax(result_logits, 1)

with tf.Session() as sess:
    
    saver.restore(sess, load_file)
    
    pred_class = sess.run(y_pred_class, feed_dict={model_x: x_new_test, keep_prob: 1.0})
    top5_prob = sess.run(top5_pred_prob, feed_dict={model_x: x_new_test, keep_prob: 1.0})

for i in range(7):
    print("Image " + str(i+1) + " belongs to class " + str(pred_class[i]) + " - " + class_info[str(pred_class[i])]['description'])
```

    Image 1 belongs to class 12 - Priority road
    Image 2 belongs to class 40 - Roundabout mandatory
    Image 3 belongs to class 1 - Speed limit (30km/h)
    Image 4 belongs to class 26 - Traffic signals
    Image 5 belongs to class 18 - General caution
    Image 6 belongs to class 16 - Vehicles over 3.5 metric tons prohibited
    Image 7 belongs to class 38 - Keep right


### Question 7

_Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate._

_**NOTE:** You could check the accuracy manually by using `signnames.csv` (same directory). This file has a mapping from the class id (0-42) to the corresponding sign name. So, you could take the class id the model outputs, lookup the name in `signnames.csv` and see if it matches the sign from the image._


**Answer:**

Data above shows that the model correctly predicted 4 out of 7 images giving  57% accuracy. It correctly predicted tehe images that were already in the class list. It was already impossible to predict the remaining 3 as there were not in the class list.


```python
### Visualize the softmax probabilities here.

for i in range(7):
    plt.imshow(x_new_test[i].squeeze(), cmap = "gray")
    plt.show()
    print("Top 5 classes with probabilities")
    for j in range(5):
        print("Class #" + str(top5_prob[1][i][j]) + 
              " - Probability: " + str(round(top5_prob[0][i][j],3)) +
              " - Description: " + class_info[str(top5_prob[1][i][j])]['description'])
```


![png](output_44_0.png)


    Top 5 classes with probabilities
    Class #12 - Probability: 0.986 - Description: Priority road
    Class #11 - Probability: 0.004 - Description: Right-of-way at the next intersection
    Class #40 - Probability: 0.004 - Description: Roundabout mandatory
    Class #42 - Probability: 0.002 - Description: End of no passing by vehicles over 3.5 metric tons
    Class #38 - Probability: 0.001 - Description: Keep right



![png](output_44_2.png)


    Top 5 classes with probabilities
    Class #40 - Probability: 0.438 - Description: Roundabout mandatory
    Class #17 - Probability: 0.339 - Description: No entry
    Class #20 - Probability: 0.067 - Description: Dangerous curve to the right
    Class #37 - Probability: 0.052 - Description: Go straight or left
    Class #12 - Probability: 0.021 - Description: Priority road



![png](output_44_4.png)


    Top 5 classes with probabilities
    Class #1 - Probability: 0.972 - Description: Speed limit (30km/h)
    Class #2 - Probability: 0.011 - Description: Speed limit (50km/h)
    Class #4 - Probability: 0.009 - Description: Speed limit (70km/h)
    Class #5 - Probability: 0.004 - Description: Speed limit (80km/h)
    Class #0 - Probability: 0.001 - Description: Speed limit (20km/h)



![png](output_44_6.png)


    Top 5 classes with probabilities
    Class #26 - Probability: 0.908 - Description: Traffic signals
    Class #24 - Probability: 0.031 - Description: Road narrows on the right
    Class #22 - Probability: 0.023 - Description: Bumpy road
    Class #37 - Probability: 0.013 - Description: Go straight or left
    Class #18 - Probability: 0.01 - Description: General caution



![png](output_44_8.png)


    Top 5 classes with probabilities
    Class #18 - Probability: 0.647 - Description: General caution
    Class #37 - Probability: 0.249 - Description: Go straight or left
    Class #33 - Probability: 0.043 - Description: Turn right ahead
    Class #25 - Probability: 0.034 - Description: Road work
    Class #20 - Probability: 0.011 - Description: Dangerous curve to the right



![png](output_44_10.png)


    Top 5 classes with probabilities
    Class #16 - Probability: 0.984 - Description: Vehicles over 3.5 metric tons prohibited
    Class #40 - Probability: 0.006 - Description: Roundabout mandatory
    Class #42 - Probability: 0.005 - Description: End of no passing by vehicles over 3.5 metric tons
    Class #41 - Probability: 0.002 - Description: End of no passing
    Class #12 - Probability: 0.001 - Description: Priority road



![png](output_44_12.png)


    Top 5 classes with probabilities
    Class #38 - Probability: 1.0 - Description: Keep right
    Class #25 - Probability: 0.0 - Description: Road work
    Class #20 - Probability: 0.0 - Description: Dangerous curve to the right
    Class #36 - Probability: 0.0 - Description: Go straight or right
    Class #42 - Probability: 0.0 - Description: End of no passing by vehicles over 3.5 metric tons


### Question 8

*Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)*

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

**Answer:**

Model yielded high probabilities for the correct predictions  - above 95%
For incorrect predictions the distribution was more uniform among classes.

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.


```python

```
