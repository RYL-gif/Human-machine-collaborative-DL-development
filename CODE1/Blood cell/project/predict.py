# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

import pickle
import cv2
import os
import numpy as np
import pandas as pd

import json
import keras
import efficientnet.keras as efn

from tensorflow.keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import img_to_array
from keras_classification_models.keras import Classifiers


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

model_name = 'inceptionv3' # resnet50  inceptionv3  effientnetB4
model_save_name = '{}_best_model.h5'.format(model_name)
savepath = './logs/{}/'.format(model_name)
test_root = "../confirm"  # testing dataset    confirm

# white_txt = "test.pickle"
resize = 299 #388
n_classes = 4
threshold = 0.5

multi_cls = True
exter_no_label = False
inceptionresnetv2_load = False


name = os.path.join('./results', '{}'.format(model_name), 'external')
if not os.path.exists(name):
	os.makedirs(name)
# if not os.path.exists(os.path.join(name, 'external')):
# 	os.path.exists(os.path.join(name, 'external'))

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

if not inceptionresnetv2_load:
	if model_name == 'effientnetB4':
		base_model = efn.EfficientNetB4(input_shape=(resize, resize, 3), weights='imagenet', include_top=False)
	else:
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
else:
	model = load_model('model.h5')


files = []
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
		files.append(each[0].split('/')[-1])
	except:
		print(each[0])

preds = np.array(preds)
pred_labels = preds.copy()
if not multi_cls:
	pred_labels[pred_labels > threshold] = 1
	pred_labels[pred_labels <= threshold] = 0
if multi_cls:
	pred_labels_ = pred_labels.tolist()
	pred_labels = np.argmax(pred_labels_,axis=1)

if not exter_no_label:
	test_cm = confusion_matrix(labels,pred_labels)
	test_auc = metrics.roc_auc_score(labels, preds, average='macro', multi_class='ovr')
	print(test_cm)

	acc = (test_cm[2, 2] + test_cm[1, 1] + test_cm[0, 0] + test_cm[3, 3]) / np.sum(test_cm)
	print('ACC:' + str(acc))
	print('AUC:' + str(test_auc))

	test_result = {
		'auc': round(test_auc, 4),
		'acc': round(acc, 4),
		'confusion matrix': '[[{}, {}, {},{}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]]'.format(test_cm[0, 0], test_cm[0, 1],test_cm[0, 2],test_cm[0, 3],
																				test_cm[1, 0], test_cm[1, 1],test_cm[1, 2],test_cm[1, 3],
																				test_cm[2, 0], test_cm[2, 1],test_cm[2, 2],test_cm[2, 3],
				                                                                test_cm[3, 0], test_cm[3, 1], test_cm[3, 2], test_cm[3, 3])
	}


	if not os.path.exists(os.path.join(name, 'test_auc{}'.format(str(round(test_auc, 2))))):
		os.makedirs(os.path.join(name, 'test_auc{}'.format(str(round(test_auc, 2)))))
	with open(os.path.join(name, 'test_auc{}'.format(str(round(test_auc, 2))), 'test_results.json'),
			  'w') as f:
		json.dump(test_result, f, indent=4)

	# if not os.path.exists(os.path.join(name, 'external_test')):
	# 	os.makedirs(os.path.join(name, 'external_test'))

	scores = pd.DataFrame(columns=['Image', 'Score0', 'Score1','Score2','Score3','Pred_Label'])
	scores['Image'] = files
	scores['Score0'] = np.around(preds[:, 0], decimals=4)
	scores['Score1'] = np.around(preds[:, 1], decimals=4)
	scores['Score2'] = np.around(preds[:, 2], decimals=4)
	scores['Score3'] = np.around(preds[:, 3], decimals=4)
	scores['Pred_Label'] = pred_labels
	# scores.to_csv(os.path.join(os.path.join(name, 'external_test', 'scores.txt')), sep=',',index=False)
	scores.to_csv(
		os.path.join(os.path.join(os.path.join(name, 'test_auc{}'.format(str(round(test_auc, 2)))), 'scores.txt')),
		sep=',', index=False)
else:
	preds = preds.tolist()
	pred_labels = pred_labels.tolist()

	name = os.path.join('./results', '{}'.format(model_name))
	if not os.path.exists(name):
		os.makedirs(name)
	if not os.path.exists(os.path.join(name, 'external_test')):
		os.makedirs(os.path.join(name, 'external_test'))

	scores = pd.DataFrame(columns=['Image', 'Score', 'Pred_Label'])
	scores['Image'] = files
	scores['Score'] = np.around(preds, decimals=4)
	scores['Pred_Label'] = pred_labels
	scores.to_csv(os.path.join(os.path.join(name, 'external_test', 'scores.txt')), sep=',',index=False)


print('saving is done')

# with open(white_txt,"wb") as f:
# 	pickle.dump(index,f)
# print(index)
