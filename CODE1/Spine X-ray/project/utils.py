import cv2
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

def image_data_generator_stacking(txt_file, batch_size, resize, df, y_train):
    train_lengh = len(txt_file)
    columns = ['mean_area',
               '性别_0','性别_1','居住类型_0','居住类型_1','居住类型_2','居住类型_3','现住址-市_0','现住址-市_1','现住址-市_2',
               'age_0','age_1','age_2','age_3','age_4','edu_0','edu_1','edu_2','job_0','job_1','job_2','job_3',
               'description_0','description_1','history_0','history_1','anti_immun_0','BMI_cls_0','BMI_cls_1',
               'BMI_cls_2','anti_immun_1','right_eye_his_0','right_eye_his_1','lf_eye_his_0','lf_eye_his_1']
    while True:
        for start in range(0, train_lengh, batch_size):
            x_batch_img = []
            x_batch_metadata = []
            end = min(start + batch_size, train_lengh)
            #x_batch = np.array([open_image(path, resize) for path in txt_file[start:end]])
            for path in txt_file[start:end]:
                id = path.split('/')[-1].split('-')[0]
                #age = df[df.ID == id]['st_age'].iloc[0]
                meta_data = df[df.ID == id][columns].values.reshape(-1)

                #x_batch.append([np.array(open_image(path, resize)), [gender,age]])
                x_batch_img.append(np.array(open_image(path, resize)))
                x_batch_metadata.append(meta_data)

            x_batch_img = np.array(x_batch_img)
            x_batch_metadata = np.array(x_batch_metadata)
            x_batch = [x_batch_img, x_batch_metadata]
            y_batch = y_train[start : end]


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

if __name__ == '__main__':
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('./results/densenet201/test_auc0.5/scores.txt')
    test_files = open('./data/external4slide_test.txt').read()
    test_files = list(test_files.split('\n'))
    names = set(df['svs_file'])

    dict = {}
    for path in test_files:
        name = path.split('/')[-2]
        label = path.split('/')[2]
        if dict.get(name) is None:
            dict[name] = label
    # svs_name = 'TCGA-B0-4842-01Z-00-DX1.d780158b-81c2-4ac9-b7a0-9c9386c6414c'
    svs_list,labels_list,quat_list = [],[],[]
    for svs_name in names:
        labels = df[df['svs_file']==svs_name]['Label']
        scores =  df[df['svs_file']==svs_name]['Score']
        # labels = df['Label']
        # scores =  df['Score']

        fig = plt.figure()
        x = scores
        ax = fig.add_subplot(111)
        numBins = 100
        ax.hist(x, numBins, color='blue', alpha=0.8, rwidth=0.9)
        # plt.grid(True)
        # plt.title(u'scores')
        # plt.show()

        quat = np.percentile(scores, (75, 90, 95, 99), interpolation='midpoint')
        print(svs_name + ' label: ' + dict[svs_name])
        print(quat)
        svs_list.append(svs_name)
        labels_list.append(dict[svs_name])
        quat_list.append(quat)

    scores = pd.DataFrame(columns=['svs_file', 'Label', 'Quat'])
    scores['svs_file'] = svs_list
    scores['Label'] = labels_list
    scores['Quat'] = quat_list

    scores.to_csv(os.path.join('./', 'yuzhi.txt'),
                  sep=',',
                  index=False)

#     save_roc(labels, preds, title='densenet201', name='./test_auc0.97')