import cv2
import numpy as np
import os
class PicReader:
    def __init__(self):
        pass

    def Reader(self,path):
        try:
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            arr = np.array(img)
        except Exception as e:
            print(e)
        tag = path.split("_")[-2].split("\\")[-1]
        return arr


if __name__ == "__main__":
    pr = PicReader()
    dir = os.getcwd() +"\\Data\\DB1\\101_1.tif"
    print(dir)
    pr.Reader(dir)