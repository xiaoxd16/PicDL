WIDTH = 160
HEIGHT = 120
from torch.utils.data import  Dataset
import torch
import os,cv2,random,numpy as np
label_dict = {}


def AllFiles(dirPath):
    output = []
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                output.append(os.path.join(root, file))
    return output

def DefaultLoader(path):
    img = cv2.imread(path,0)
    label = path.split('_')[0]
    return img,label

class TrainSet(Dataset):
    files = []
    size = 800
    right_size = 200
    idxs = None
    def __init__(self,dirPath='Data/DB1',size = None):
        self.files = []
        self.AddFiles(dirPath)
        if size is not None:
            self.size = size
        self.Random()

    def Random(self):
        self.idxs = []
        i = 0
        right = 0
        while i < self.size:
            r1 = random.randint(0, len(self.files) - 1)
            r2 = random.randint(0, len(self.files) - 1)
            label1 = self.files[r1].split('_')[0]
            label2 = self.files[r2].split('_')[0]
            if (right < self.right_size and label2 != label1):
                continue
            self.idxs.append((r1,r2))
            if (right < self.right_size and label2 == label1):
                right += 1
            i+=1
        np.random.shuffle(self.idxs)

    def AddFiles(self,dirPath):
        fs = AllFiles(dirPath)
        self.files += fs
        self.Random()

    def __getitem__(self, index):
        id1,id2 = self.idxs[index]
        f1 = self.files[id1]
        f2 = self.files[id2]
        img1, label1 = DefaultLoader(f1)
        img2, label2 = DefaultLoader(f2)
        img1 = img1.reshape((1,)+img1.shape)
        img2 = img2.reshape((1,) + img2.shape)
        data1 = torch.from_numpy(img1).float() / 255
        data2 = torch.from_numpy(img2).float() / 255
        label = torch.from_numpy(np.array(label1 == label2)).float()
        return (data1,data2),label

    def __len__(self):
        return self.size

    def GenerateOutput(self):
        imgs = []
        output = None
        labels = None
        for i in range(len(self.files)):
            img,label = self.__getitem__(i)
            imgs.append((img,label))
        for i in range(self.size):
            r1 = random.randint(0,len(self.files) - 1)
            r2 = random.randint(0, len(self.files) - 1)
            img1,l1 = imgs[r1]
            img2,l2 = imgs[r2]
            l = (l1 == l2)
            d = np.stack([img1,img2] ,axis=0)
            if output is None:
                output = d.reshape(((1,)+d.shape))
                labels = np.array(l).reshape((-1,1))
            else:
                output = np.insert(output,-1,values=d,axis=0)
                labels = np.insert(labels,-1,values=l,axis=0)
        return output,labels



if __name__ == '__main__':
    s = TrainSet()
    s.AddFiles('Data/DB2')
    s.AddFiles('Data/DB3')
    a,l = s[0],s[1]
    print(l)
