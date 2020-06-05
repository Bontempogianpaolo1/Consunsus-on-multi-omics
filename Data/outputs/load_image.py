from matplotlib import pyplot
import cv2
import numpy as np

import os


models=["bnn","mlp","svm","random-forest"]
filenames = ["meth","mrna","micro mrna"]
path="./"
for model in models:
    imgs=list()
    for filename in filenames:
        a=cv2.imread(path+model+"-"+filename+".png")
        imgs.append(a)
    imgs.append(cv2.imread(path+"comparison"+model+".png"))
    im_v = cv2.hconcat([imgs[0],imgs[1],imgs[2],imgs[3]])
    cv2.imwrite('../report/'+model+'.jpg', im_v)

