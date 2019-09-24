import cv2 as cv
import numpy as np

import math
import sys
from numpy.linalg import inv


class Sparse:

    def __init__(self, video, matrix):
        self.video = video
        self.matrix = matrix

    def inverseTransform(self, a, b):
        m = inv(self.matrix)
        src = np.array([[a, b]], dtype=np.float32)
        src = np.array([src])
        transf = cv.perspectiveTransform(src, m)
        a = transf[0][0][0]
        b = transf[0][0][1]

        return (a, b)

    def evaluate(self, feature_params, lk_params, x, y):
        
        color = (0, 0, 255)

        first_frame = self.video[0]
        prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
        
        ct_first_frame = cv.warpPerspective(first_frame, self.matrix, (int(x), int(y)))
        ct_prev_gray = cv.cvtColor(ct_first_frame, cv.COLOR_BGR2GRAY)
        prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
        ct_prev = cv.goodFeaturesToTrack(ct_prev_gray, mask = None, **feature_params)

        mask = np.zeros_like(ct_first_frame)
        ct_mask = np.zeros_like(ct_first_frame)

        

        
            
        ct_frame = cv.warpPerspective(self.video[1], self.matrix, (int(x), int(y)))
        
        
        gray = cv.cvtColor(self.video[1], cv.COLOR_BGR2GRAY)

        ct_gray = cv.cvtColor(ct_frame, cv.COLOR_BGR2GRAY)
        next, status, error = cv.calcOpticalFlowPyrLK(ct_prev_gray, ct_gray, ct_prev, None, **lk_params)
        
        good_old = ct_prev[status == 1]
        good_new = next[status == 1]

        movements = []
        untransformed = []

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            
            a, b = new.ravel()
            c, d = old.ravel()

            dotdistance = math.sqrt(abs(math.pow((a-c),2) + math.pow((b-d),2)))
            #print(dotdistance)
            
            movements.append((self.inverseTransform(a, b), self.inverseTransform(c, d)))
            untransformed.append(((a, b), (c, d)))
            
        return (untransformed, movements)
        
            