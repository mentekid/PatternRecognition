# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:46:01 2016

@author: et3rn1ty
"""
import matplotlib.pyplot as pl

import data_read as dtrd
data_const, features, projects_true = dtrd.All(sparse=False)

import mypreprocessing as prp
data_const = prp.TrimmingPresence(data_const, low_thresh=3)
from mvpa2.suite import *

#colors = np.array(
#         [[0., 0., 0.],
#          [0., 0., 1.],
#          [0., 0., 0.5],
#          [0.125, 0.529, 1.0],
#          [0.33, 0.4, 0.67],
#          [0.6, 0.5, 1.0],
#          [0., 1., 0.],
#          [1., 0., 0.],
#          [0., 1., 1.],
#          [1., 0., 1.],
#          [1., 1., 0.],
#          [1., 1., 1.],
#          [.33, .33, .33],
#          [.5, .5, .5],
#          [.66, .66, .66]])

colors = data_const

# store the names of the colors for visualization later on
color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']
         
som = SimpleSOMMapper((20, 30), 400, learning_rate=0.05)
som.train(colors)


pl.imshow(som.K, origin='lower')
mapped = som(colors)

pl.title('Color SOM')
# SOM's kshape is (rows x columns), while matplotlib wants (X x Y)
for i, m in enumerate(mapped):
    pl.text(m[1], m[0], color_names[i], ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.5, lw=0))
pl.show()