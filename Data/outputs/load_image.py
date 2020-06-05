from matplotlib import pyplot
import cv2
import numpy as np

import os



path="./paretos/"
modello="with feature selection"
a1 = cv2.imread(path+"preprocessed-meth svm.png")
a2 = cv2.imread(path+"preprocessed-mrna Random Forest.png")
a3 = cv2.imread(path+"preprocessed-micro mrna  Random Forest.png")
#a4 = cv2.imread("comparison svm .png")
im_v = cv2.hconcat([a3, a2,a1])
#im_v = cv2.hconcat([im_v, a3])
cv2.imwrite(modello+'.jpg', im_v)

modello="without feature selection"
a1 = cv2.imread(path+"meth svm.png")
a2 = cv2.imread(path+"mrna Random Forest.png")
a3 = cv2.imread(path+"micro mrna  Random Forest.png")
#a4 = cv2.imread("comparison svm .png")
im_v = cv2.hconcat([a3, a2,a1])
#im_v = cv2.hconcat([im_v, a3])
cv2.imwrite(modello+'.jpg', im_v)
