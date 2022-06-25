import os
import matplotlib.image as mpimg
from torch.utils import data
from PIL import Image
from torchvision import transforms as tfs
import torch
def train_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize((224,224)),
        tfs.CenterCrop(224),
        #tfs.Pad(1), #512 match 514
        #tfs.Grayscale(num_output_channels = 1),#胸片里掺了灰度图进去
        tfs.ToTensor(),
        #tfs.Normalize([0.5], [0.5])
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ])
    #print("x.shape",x.shape)
    x = im_aug(x)
    #print("im_aug(x)",x.shape)
    return x

def Get_Set(root):
    # 所有图片的绝对路径,clas0开始的类名，clas1为最后一个类名
    root_path = os.listdir(root)
    imgs = []
    val = []
    vals = []
    for c in root_path:
        # 读取分类文件夹
        c_path = os.listdir(root + '/' + c)
        val.append(c)
        for img in c_path:
            # 读取图片地址
            img_path = root + '/' + c + "/" + img
            pil_img = Image.open(img_path)
            # 将图片按自定义的transforms格式转换
            #data = mpimg.imread(img_path)  # 直接读取成numpy格式
            #data = data.flatten()  # 将numpy展平
            data = train_tf(pil_img)
            # 将图片压缩成1维
            data = torch.flatten(data)
            imgs.append(data.numpy())
            vals.append(c)

    return imgs, val, vals
