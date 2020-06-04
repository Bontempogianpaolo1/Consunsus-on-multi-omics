from matplotlib import pyplot
import cv2
import numpy as np

import os
im_mlp=cv2.vconcat([np.ndarray([])])
im_random=cv2.vconcat([np.ndarray([])])
im_svm=cv2.vconcat([np.ndarray([])])
im_bnn=cv2.vconcat([np.ndarray([])])
filelist=os.listdir('./')
'''
for fichier in filelist:
    if(fichier.endswith(".png") and fichier.startswith("mlp")):
        a1=cv2.imread(fichier)
        im_v = cv2.vconcat([im_mlp, a1])
    if (fichier.endswith(".png") and fichier.startswith("random")):
        a1 = cv2.imread(fichier)
        im_v = cv2.vconcat([im_random, a1])
    if (fichier.endswith(".png") and fichier.startswith("svm")):
        a1 = cv2.imread(fichier)
        im_v = cv2.vconcat([im_svm, a1])
    if (fichier.endswith(".png") and fichier.startswith("bnn")):
        a1 = cv2.imread(fichier)
        im_v = cv2.vconcat([im_bnn, a1])'''
modello="mlp"
a1 = cv2.imread(modello+" meth.png")
a2 = cv2.imread(modello+" mrna.png")
a3 = cv2.imread(" "+modello+" mrna normalized.png")
a4 = cv2.imread("comparison "+modello+" .png")
im_v = cv2.hconcat([a3, a2,a1,a4])
#im_v = cv2.hconcat([im_v, a3])
cv2.imwrite(modello+'.jpg', im_v)

modello="svm"
a1 = cv2.imread(" svm meth.png")
a2 = cv2.imread(" svm mrna.png")
a3 = cv2.imread(" svm mrna normalized.png")
a4 = cv2.imread("comparison svm .png")
im_v = cv2.hconcat([a3, a2,a1,a4])
#im_v = cv2.hconcat([im_v, a3])
cv2.imwrite(modello+'.jpg', im_v)
modello="random-forest"
a1 = cv2.imread(" random-forest meth.png")
a2 = cv2.imread(" "+modello+" mrna.png")
a3 = cv2.imread(" "+modello+" mrna normalized.png")
a4 = cv2.imread("comparison "+modello+" .png")
im_v = cv2.hconcat([a3, a2,a1,a4])
#im_v = cv2.hconcat([im_v, a3])
cv2.imwrite(modello+'.jpg', im_v)
modello="bnn"
a1 = cv2.imread(modello+" meth.png")
a2 = cv2.imread(modello+" mrna.png")
a3 = cv2.imread("bnn micro-rna.png")
a4 = cv2.imread("bnn finale.png")
im_v = cv2.hconcat([a3, a2,a1,a4])
#im_v = cv2.hconcat([im_v, a3])
cv2.imwrite(modello+'.jpg', im_v)
