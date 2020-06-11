from matplotlib import pyplot
import cv2
import numpy as np

import os


models=["mlp-local-outlier","mlp-one-class","mlp-isolation-forest","mlp-elliptic"]
filenames = ["meth","mrna","micro mrna"]
path="./"
for model in models:
    imgs=list()
    for filename in filenames:
        a=cv2.imread(path+model+filename+"pca.png")
        imgs.append(a)
    im_v = cv2.hconcat([imgs[0], imgs[1], imgs[2]])
    cv2.imwrite('../report/'+model+'.jpg', im_v)
imgs=list()
for model in models:


    a=cv2.imread('../report/'+model+'.jpg')
    imgs.append(a)
im_v = cv2.vconcat([imgs[0], imgs[1], imgs[2],imgs[3]])
cv2.imwrite('../report/matrix_pca_on_test_set.jpg', im_v)


models=["newdata-mlp-local-outlier","newdata-mlp-one-class","newdata-mlp-isolation-forest","newdata-mlp-elliptic"]
filenames = ["meth","mrna","micro mrna"]
path="./"
for model in models:
    imgs=list()
    for filename in filenames:
        a=cv2.imread(path+model+"-"+filename+"pca.png")
        imgs.append(a)
    im_v = cv2.hconcat([imgs[0], imgs[1], imgs[2]])
    cv2.imwrite('../report/'+model+'new_data.jpg', im_v)
imgs=list()
for model in models:


    a=cv2.imread('../report/'+model+'new_data.jpg')
    imgs.append(a)
im_v = cv2.vconcat([imgs[0], imgs[1], imgs[2],imgs[3]])
cv2.imwrite('../report/matrix_pca_on_new_data.jpg', im_v)
