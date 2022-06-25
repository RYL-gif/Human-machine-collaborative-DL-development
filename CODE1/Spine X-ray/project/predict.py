# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

import pickle
import cv2
import os
import numpy as np

import json
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import img_to_array
from keras_classification_models.keras import Classifiers



# img_root = "D:\\hotmap\\tiff256_5"  #需要判断的图片
# img_root = "D:\\ai-path-structraining-1129\\data_withe\\white"
          #复制到的地址
           #模型文件夹

model_name = 'inceptionv3' # resnet50  inceptionv3
model_save_name = '{}_best_model.h5'.format(model_name)
savepath = './logs/{}/'.format(model_name)
test_root = "../testing dataset"
white_txt = "test.pickle"
resize = 299
n_classes = 2
threshold = 0.5
multi_cls = False

def getImagePathList(dataset_dir):
    imagePathList = []
    print("ok")
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            imagePathList.append(os.path.join(root, file))
    print("got_list")
    print(len(imagePathList))
    return imagePathList

#In[]
r_path_list = os.listdir(test_root)
path_list = []
for each in r_path_list:
	path_list.append(os.path.join(test_root,each))
imagePaths = []
for i,each_class in enumerate(path_list):
	file_list = getImagePathList(each_class)
	for each in file_list:
		imagePaths.append([each,i])

# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")





base_net, preprocess_input = Classifiers.get(model_name.lower())
base_model = base_net(input_shape=(resize, resize, 3), weights='imagenet', include_top=False)
# build model top
x = keras.layers.GlobalAveragePooling2D()(base_model.output)

# classification output
if not multi_cls:
	output = keras.layers.Dense(1, activation="sigmoid")(x)
else:
	output = keras.layers.Dense(n_classes, activation="softmax")(x)

model = keras.models.Model(inputs=[base_model.input], outputs=[output])
savepath = os.path.join(savepath,model_save_name)
model.load_weights(savepath,by_name=True)


index = []
preds, labels = [],[]
for i,each in enumerate(imagePaths):
	print("[INFO] classifying image..." ,i,"/",len(imagePaths))
	try:
		image = cv2.imread(each[0])
		# output = imutils.resize(image, width=400)

		# pre-process the image for classification
		image = cv2.resize(image, (resize, resize))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		proba = model.predict(image)[0]

		preds.append(proba)
		labels.append(int(each[1]))
		index.append([each[0],each[1],proba])
	except:
		print(each[0])

preds = np.array(preds)
if not multi_cls:
    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0
if multi_cls:
    labels = np.argmax(labels,axis=1)
test_cm = confusion_matrix(labels,preds)

test_auc = metrics.roc_auc_score(labels, preds, average='macro')

sensitivity = test_cm[1,1]/(test_cm[1,1] + test_cm[1,0])
specificity = test_cm[0,0]/(test_cm[0,0] + test_cm[0,1])
acc = (test_cm[1, 1] + test_cm[0, 0]) / np.sum(test_cm)
print('sensitivity: '+ str(sensitivity))
print('specificity: '+ str(specificity))
print('ACC:' + str(acc))
print('AUC:' + str(test_auc))

test_result = {
	'auc': round(test_auc, 4),
	'acc': round(acc, 4),
	'sensitivity': round(sensitivity, 4),
	'specificity': round(specificity, 4),
	'confusion matrix': '[[{}, {}], [{}, {}]]'.format(test_cm[0, 0], test_cm[0, 1], test_cm[1, 0], test_cm[1, 1]),
    }

name = os.path.join('./results', '{}'.format(model_name))
if not os.path.exists(name):
	os.makedirs(name)
if not os.path.exists(os.path.join(name, 'test_auc{}'.format(str(round(test_auc, 2))))):
	os.makedirs(os.path.join(name, 'test_auc{}'.format(str(round(test_auc, 2)))))
with open(os.path.join(name, 'test_auc{}'.format(str(round(test_auc, 2))), 'test_results.json'),
		  'w') as f:
	json.dump(test_result, f, indent=4)

print('saving is done')

# with open(white_txt,"wb") as f:
# 	pickle.dump(index,f)
# print(index)
