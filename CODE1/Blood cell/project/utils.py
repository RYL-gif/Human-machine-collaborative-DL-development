import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
#import tensorflow as tf
from skimage import exposure
from keras.utils.np_utils import to_categorical
# from augmentation_pipeline import aug_pipeline
import imgaug.augmenters as iaa


#def auc(y_true, y_pred):
#  return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def save_roc(targets, scores, title, name):
    plt.figure()
    fpr, tpr, thresholds = roc_curve(targets, scores)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

    plt.plot(fpr, tpr,color='steelblue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    plt.text(optimal_point[0], optimal_point[1], 'Threshold:{optimal_th:.2f}')
    #plt.plot(fpr, tpr, color='steelblue', label='{} (AUC={},{}-{})'.format('Hepatobiliary disease', roc_auc,'0·65','0·71'))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(loc='lower right',prop = {'size':7.5})
    #plt.legend(loc='best')
    plt.savefig(name + '.tiff')


def cal_confidence_interval(value, n, inter_value):
    if inter_value == 0.9:
        const = 1.64
    elif inter_value == 0.95:
        const = 1.96
    elif inter_value == 0.98:
        const = 2.33

    confidenc_interval_upper = value + const * np.sqrt((value * (1 - value)) / n )
    confidenc_interval_lower = value - const * np.sqrt((value * (1 - value)) / n )
    if confidenc_interval_lower < 0:
        confidenc_interval_lower = 0
    return (round(confidenc_interval_lower,4),round(confidenc_interval_upper,4))

def open_image(path, resize):
    #print(path)
    if path == '':
        return
    path = path.split(',')[0]
    #x = cv2.imread(path)
    x = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if x is None:
        print('error path: ' + path)

    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (resize, resize))
    # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # x = [x] * 3
    # x = cv2.merge(x)

    # Normalization 0-1
    x = x / 255.
    return x


def image_data_generator(txt_file, batch_size, resize, y_train):

    train_lengh = len(txt_file)
    while True:
        for start in range(0, train_lengh, batch_size):
            end = min(start + batch_size, train_lengh)
            x_batch = np.array([open_image(path, resize) for path in txt_file[start:end]])
            y_batch = np.array(y_train[start : end])

            yield x_batch, y_batch


def TTA_image_data_generator(txt_file, resize, aug_times):
    pipeline = aug_pipeline()
    aug = pipeline.augmentation()
    aug_imgs_list = []

    for path in txt_file:
        if path == '':
            return
        path = path.split(',')[0]
        # x = cv2.imread(path)
        x = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        if x is None:
            print('error path: ' + path)
        # img = cv2.imread(file)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (resize, resize))
        k = 0
        #aug_imgs = []
        while k < aug_times:
            aug_img = aug.augment_image(x)
            aug_img = aug_img / 255.
            aug_imgs_list.append(aug_img)
            k += 1
        #aug_imgs_list.append(aug_imgs)

    aug_imgs_list = np.array(aug_imgs_list)
    return aug_imgs_list

class data_generator:

    def __init__(self, root_path, _max_example, BATCHSIZE, image_size, classes, word_id):
        self.root_path = root_path
        self.index = 0
        self.batch_size = BATCHSIZE
        self.image_size = image_size
        self.classes = classes
        self.num_of_examples = _max_example
        self.word_id = word_id
        self.load_images_labels(_max_example)


    def getImagePathList(self, dataset_dir):
        imagePathList = []
        print("ok")
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                imagePathList.append(os.path.join(root, file))
        print("got_list")
        print(len(imagePathList))
        return imagePathList

    def load_images_labels(self, _max_example):
        r_path_list = os.listdir(self.root_path)
        path_list = []
        for each in r_path_list:
            path_list.append(os.path.join(self.root_path, each))
        imagePaths = []
        for i, each_class in enumerate(path_list):
            file_list = self.getImagePathList(each_class)
            label = each_class.split('/')[-1]
            label2ind = self.word_id[label]
            for each in file_list:
                imagePaths.append([each, label2ind])

        images = []
        labels = []
        random.shuffle(imagePaths)
        for i, each in enumerate(imagePaths):
            print("[INFO] loading image...", i, "/", len(imagePaths))
            try:
                image = cv2.imread(each[0])
                # output = imutils.resize(image, width=400)

                # pre-process the image for classification
                image = cv2.resize(image, (self.image_size, self.image_size))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype("float") / 255.0
                # image = image / 255.
                image = np.array(image)
                images.append(image)

                label = each[1]
                labels.append(label)

            except:
                print(each[0])

        y_train = to_categorical(labels, num_classes=None)
        self.images = images
        self.labels = y_train


    def get_mini_batch(self):
        while True:
            batch_images = []
            batch_labels = []
            for i in range(self.batch_size):
                if (self.index == len(self.images)):
                    self.index = 0
                batch_images.append(self.images[self.index])
                batch_labels.append(self.labels[self.index])
                self.index += 1
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            yield batch_images, batch_labels