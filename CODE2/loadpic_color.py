import os
from torch.utils import data
from PIL import Image
from torchvision import transforms as tfs

def train_tf(x):
    im_aug = tfs.Compose([
        tfs.Resize((298,298)),
        tfs.CenterCrop(298),
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


class Get_TrainSet(data.Dataset):
    #读取图片及其标签
    def __init__(self,root):
        # 所有图片的绝对路径
        root_path=os.listdir(root)
        self.imgs = []
        self.val = []
        for c in root_path:
            #读取分类文件夹
            c_path = os.listdir(root+'\\'+c)
            for img in c_path:
                #读取图片地址
                self.imgs.append(root+'\\'+c+"\\"+img)
                self.val.append(c)


    def __getitem__(self, index):
        img_path = self.imgs[index]
        val = self.val[index]
        #打开图片
        pil_img = Image.open(img_path)
        data = train_tf(pil_img)

        return data,val

    def __len__(self):
        #数据集所有图片数量
        return len(self.imgs)
