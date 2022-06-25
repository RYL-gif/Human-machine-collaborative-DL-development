from loadpic_color import Get_TrainSet
from Autoencoder import AutoEncoder_CNN
from julei import julei
import torch
from torch import nn
import numpy as np
import itertools as it
import copy
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import models


julei_fn = ["kmeans", "cmeans", "GMM"]
#julei_fn = ["kmeans", "kmeans","kmeans"]
import os

def eval_model(julei_fn,model):
    dataset = []
    vals = []
    label = []
    for img, val in testloader:
        img = img.cuda()
        en = model(img)
        en=en.detach().cpu().numpy()
        #print(en.shape)
        '''
        #输出原始图片三通道
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(img[0][i].cpu().numpy(),cmap='gray')
        plt.show()
        for i in range(16):
            #输出卷积后图片
            plt.subplot(4, 4, i + 1)
            #print(en[0][i].shape)
            plt.imshow(en[0][i], cmap='gray')
            #plt.axis('off')
        plt.show()
        '''
        dataset.append(np.array(en.flatten()))
        vals.append(val[0])
        if val[0] not in label:
            label.append(val[0])

    print(label)

    # 然后把dataset放到模型里，输出后再弄成np
    X = np.array(dataset)
    num_cluster = len(label)  # 聚类数量
    acclist=[]
    for times in range(10):
        #每种聚类跑10遍
        acc_list=[]
        for fn in julei_fn:
            #max_pro_val = 0
            #尝试三种聚类方式，将每种的结果保存到list，再输出到csv
        
            print(f"{fn}第{times+1}次聚类")
            choice_cluster = julei(X, num_cluster, fn)

            # 对预测值和实际标签进行全排列，取精度最高的作为实际精度
            pailie = []
            all_cluster = range(num_cluster)  # 聚的类[0,1,2]
            # 可以直接对all_cluster排列组合，然后再和label组
            li = list(it.permutations(all_cluster, num_cluster))
            # print(li)
            newpailie = []
            for j in li:
                for i in all_cluster:
                    pailie.append([j[i], label[i]])
                newpailie.append(copy.deepcopy(pailie))
                pailie = []
            # print(newpailie)

            allnum = len(vals)  # 样本总数
            acc_dic = {}
            # 计算每种组合的精度
            for ii in newpailie:
                # 每一种排列组合
                #print(f'排列组合{ii}')
                num = 0
                for i in ii:
                    n1 = 0
                    t1 = 0
                    for index in range(allnum):
                        # choice_cluster[index]是预测值，vals[index]是真实值
                        # i[0]是cluster, i[1]是label
                        if i[1] == vals[index]:
                            t1 += 1
                            if i[0] == choice_cluster[index]:
                                n1 += 1
                                num += 1
                    each_acc = float(n1) / float(t1)
                    #print(f'类别{i}, 准确度为{n1}/{t1}, {each_acc:.2f}')
                acc = float(num) / float(allnum)
                acc_dic[str(ii)] = acc
            probco = max(acc_dic, key=acc_dic.get)
            probco_val = acc_dic[probco]
            print(f'可能性最高的组合为：{probco}，其准确度为：{probco_val * 100:.2f}%\n')
            #if probco_val>max_pro_val:
                #max_pro_val=probco_val
            acc_list.append(probco_val)
        acclist.append(copy.deepcopy(acc_list))
    '''
       plt.xlabel('')
       plt.ylabel('')
       plt.title(model.__class__.__name__ + '_kmean')

       for i in all_cluster:
           # 按聚类画点
           indices = np.where(choice_cluster == i)[0]
           # print(indices)
           selected = X[indices]
           print(selected.shape)
           # print("len:selected[:, 0]",len(selected[:, 0]))
           if i == 0:
               plt.plot(selected[:, 0], selected[:, 2], '.', label=str(i), color="orange")
           if i == 1:
               plt.plot(selected[:, 0], selected[:, 2], '.', label=str(i), color="blue")
           if i == 2:
               plt.plot(selected[:, 0], selected[:, 2], '.', label=str(i), color="green")
       plt.legend()
       # plt.show()
       plt.savefig('pic\\' + str(int(probco_val * 100)) + model.__class__.__name__ + '_' + fn + '.png')
       '''
    
    print("acclist",acclist)
    return acclist



#图片路径
#root = "E:\无监督训练练习\处理的图片299"
#考虑可以自动搞所有文件夹分类
root="F:\拷贝\\new\裁剪\裁剪\彩色"
#path = "E:\无监督训练练习\处理的图片（测试）"
num_epochs=30
learning_rate=0.0001
#3e-4

root_path=os.listdir(root)

for c in root_path:
    output_list = []  # [acc_kmean, acc_cmean, acc_GMM]
    print(c,'\n')
    path=root + '\\' + c

    Trainset = Get_TrainSet(path)
    testloader = torch.utils.data.DataLoader(Trainset)  #载入测试集

    model = models.resnet50(True)
    # 输出fc层之前的特征，my_model1相当于和resnet50相同的处理但是没有最后的fc层
    my_model1 = nn.Sequential(*list(model.children())[:-1])
    my_model1 = my_model1.cuda()

    output_list=eval_model(julei_fn,my_model1)


    output_numpy = np.array(output_list)
    party_df = pd.DataFrame(output_numpy)
    party_df.columns = julei_fn
    csv_name = "data\\resNet50\\"+c+"_resNet50_.csv"
    party_df.to_csv(csv_name, index=False, header=True)
