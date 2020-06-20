from matplotlib import pyplot
import cv2
import numpy as np

import os


models = ["bnn", "mlptree", "mlp"]
filenames = ["micro mrna", "meth", "mrna"]
path="./outliers-testset-"
for modelname in models:
    imgs=list()
    for filename in filenames:
        a=cv2.imread(path+modelname + "-" + filename+".png")
        imgs.append(a)

    imgs.append(cv2.imread(path+ modelname + "-comparison.png"))
    im_v = cv2.hconcat([imgs[0],imgs[1],imgs[2],imgs[3]])

    cv2.imwrite('../report/outlier-testset-'+modelname+'.jpg', im_v)

path="./outliers-stomaco-"
for modelname in models:
    imgs=list()
    for filename in filenames:
        a=cv2.imread(path+modelname + "-" + filename+".png")
        imgs.append(a)

    imgs.append(cv2.imread(path+ modelname + "-comparison.png"))
    im_v = cv2.hconcat([imgs[0],imgs[1],imgs[2],imgs[3]])

    cv2.imwrite('../report/outlier-stomaco-'+modelname+'.jpg', im_v)




