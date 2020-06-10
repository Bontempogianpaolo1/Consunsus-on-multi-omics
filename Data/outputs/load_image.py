from matplotlib import pyplot
import cv2
import numpy as np

import os


models=["bnn","mlp","svm","random-forest","bnn-with-threeshold","mlp-isolation forest","isolationforest-pca","newdata-isolationforest-pca"]
filenames = ["meth","mrna","micro mrna"]
path="./"
for model in models:
    imgs=list()
    for filename in filenames:
        a=cv2.imread(path+model+"-"+filename+".png")
        imgs.append(a)
    if (model!="bnn-with-threeshold") and (model!="mlp-isolation forest") and (model!="isolationforest-pca")and (model!="newdata-isolationforest-pca"):
        imgs.append(cv2.imread(path+"comparison"+model+".png"))
        im_v = cv2.hconcat([imgs[0],imgs[1],imgs[2],imgs[3]])
    else:
        im_v = cv2.hconcat([imgs[0], imgs[1], imgs[2]])
    cv2.imwrite('../report/'+model+'.jpg', im_v)

models=["augmented","original"]
for model in models:
    imgs=list()
    imgs2=list()
    for filename in filenames:
        a1=cv2.imread(path+models[0]+filename+".png")
        a2 = cv2.imread(path + models[1] + filename + ".png")

        im_v = cv2.hconcat([a2, a1])
        cv2.imwrite('../report/vae'+filename+'.jpg', im_v)
        imgs2.append(im_v)
    #im_h = cv2.vconcat([imgs2[0], imgs2[1],imgs2[2]])



