#!/usr/bin/env python
# -*- coding: latin-1 -*-
# TestVectors.py
# Written by André Carrington
#
#    Copyright 2022 University of Ottawa
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Uses test vectors from Fawcett[1], Hilden[2], Carrington et al[3]
#
# References:
# 1. Fawcett T. An Introduction to ROC Analysis, Pattern
#    Recognition Letters, 2005.
# 2. Hilden J. The Area under the ROC Curve and Its Competitors,
#    Medical Decision Making, 1991.
# 3. Carrington AM, Fieguth P, Qazi H, Holzinger A, Chen H, Mayr F
#    Manuel D. A new concordant partial AUC and partial c statistic
#    for imbalanced data in the evaluation of machine learning
#    algorithms. BMC Medical Informatics and Decision Making, 2020.
#
import numpy as np

def getNumberOfTestVectors():
    testNum = 1
    while True:
        scores, labels, pAUCranges, groupAxis, descr = getTestVector(testNum, noError=True)
        if len(scores) == 0:
            break
        #endif
        testNum = testNum + 1
    #endwhile
    return testNum-1
#enddef

def getTestVector(testNum, noError=False):

    # all test groups are defined along FPR as the groupAxis, unless otherwise specified
    groupAxis = 'FPR'

    if testNum == 1:  # old testNum 1
        descr  = 'Test 1. Fawcett Figure 3 data (balanced classes) with partial curve boundaries ' + \
                 'aligned with instances on step verticals'
        scores = np.array([ 0.9,  0.8,  0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38,
                           0.37, 0.36, 0.35, 0.34, 0.33, 0.30, 0.1])
        labels = np.array([   1,    1,    0,   1,    1,    1,    0,    0,    1,     0,   1,    0,    1,
                              0,    0,    0,   1,    0,    1,    0])
        groups = [[0.0, 0.3], [0.3, 0.5], [0.5, 1.0]]

    elif testNum == 2:  # old testNum 2 (scores made proper for reference, no effect on measurement)
        descr  = 'Test 2. Carrington Figure 7 data (with a 1:3 P:N class imbalance) with partial curve ' + \
                 'boundaries aligned with instances.'
        # This has the same scores as Carrington Figure 8, and scores similar to Fawcett Figure 3,
        # but the labels are altered for class imbalance
        scores = [ 0.95,  0.9,  0.8,  0.7, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.5, 0.49, 0.48, 0.47,
                   0.46, 0.45, 0.44, 0.43, 0.40, 0.2]
        labels = [    1,    1,    0,    1,    1,    0,    0,    0,    0,    0,   0,    0,    1,    0,
                      0,    0,    0,    0,    0,    0]
        groups = [[0.0, 0.2], [0.2, 0.4], [0.4, 1.0]]

    elif testNum == 3:  # no old testNum equivalent
        descr  = 'Test 3. Hilden Figure 2a data with scores reversed to follow the normal convention'
        # scores (0, 1, 2, 3) high for negative were changed to (3, 2, 1, 0) respectively, high for positive
        scores = [ 3,  3,  3,  3,  3,  3,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
        labels = [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        groups = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 4:  # no old testNum equivalent
        descr = 'Test 4. Hilden Figure 2b data with scores reversed to follow the normal convention'
        # scores (0, 1, 2, 3) high for negative were changed to (3, 2, 1, 0) respectively, high for positive
        scores = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0]
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        groups = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 5:  # old testNum 5 (scores made proper for reference, no effect on measurement)
        descr  = 'Test 5. Hilden Figure 2c data with scores reversed to follow the normal convention'
        # scores (0, 1, 2, 3) high for negative were changed to (3, 2, 1, 0) respectively, high for positive
        # histogram counts were divided by 5, so that (50, 35, 15) became (10, 7, 3)
        scores = [   0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
        labels = [   0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        groups = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 6:  # old testNum 6
        descr  = 'Test 6. Fawcett Figure 3 data with partial curve boundaries aligned with instances ' + \
                 'on step horizontals'
        scores = [ 0.9,  0.8,  0.7,  0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37,
                  0.36, 0.35, 0.34, 0.33, 0.30, 0.1 ]
        labels = [   1,    1,    0,    1,    1,   1,    0,    0,    1,     0,   1,    0,    1,    0,
                     0,    0,    1,    0,    1,   0 ]
        groups = [[0.0, 0.2], [0.2, 0.6], [0.6, 1.0]]

    elif testNum == 7:  # old testNum 7
        descr  = 'Test 7. Fawcett Figure 3 data with partial curve boundaries not aligned with '\
                 'instances, requiring interpolation'
        scores = [ 0.9,  0.8,  0.7,  0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37,
                  0.36, 0.35, 0.34, 0.33, 0.30, 0.1 ]
        labels = [   1,    1,    0,    1,    1,    1,    0,    0,    1,     0,   1,    0,    1,    0,
                     0,    0,    1,    0,    1,    0]
        groups = [[0.0, 0.17], [0.17, 0.52], [0.52, 1.0]]

    elif testNum == 8:  # old testNum 8
        descr  = 'Test 8. Carrington Figure 4 data with the same shape and measure as Fawcett Figure 3 ' + \
                 'but with different scores'
        scores = [ 0.95,  0.9,  0.8,  0.7, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.5, 0.49, 0.48, 0.47,
                   0.46, 0.45, 0.44, 0.43, 0.40, 0.2]
        labels = [    1,    1,    0,    1,    1,    1,    0,    0,    1,    0,   1,    0,    1,    0,
                      0,    0,    1,    0,    1,   0]
        groups = [[0.0, 0.17], [0.17, 0.52], [0.52, 1.0]]
        #groups = [[0.0, 0.3], [0.3, 0.5], [0.5, 1.0]]
        #groups = [[0.8, 1.0]]
        #groups = [[0.4, 0.6]]
    #endif

    elif testNum == 9:  # old testNum 9
        descr  = 'Test 9. Carrington Figure 7 data: same instances as Carrington Figure 4, but different ' + \
                 'labels'
        scores = [ 0.95,  0.9,  0.8,  0.7, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.5, 0.49, 0.48, 0.47,
                   0.46, 0.45, 0.44, 0.43, 0.40, 0.2]
        labels = [    1,    1,    0,    1,    1,    0,    0,    0,    0,    0,   0,    0,    1,    0,
                      0,    0,    0,    0,    0,    0]
        groups = [[0.0, 0.17], [0.17, 0.52], [0.52, 1.0]]

    elif testNum == 10:  # old testNum 3
        descr  = 'Test 10. Carrington Figure 8 data'
        scores = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        groups = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 11:  # old testNum 4
        descr  = 'Test 11. A classifier that does worse than "continuous" chance'
        scores = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        labels = [  0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
        groups = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 12:
        descr = 'Test 12. Simplest test'
        scores = [0.8, 0.7, 0.7, 0.6]
        labels = [  1,   1,   0,   0]
        groups = [[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]]

    elif testNum == 13:
        descr = 'Test 13. A simple test'
        scores = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.7, 0.7, 0.7]
        labels = [  1,   1,   0,   1,   1,   0,   1,   1,   0]
        groups = [[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]]

    elif testNum == 14:  # previously 3b
        descr = 'Test 14.(3b) A variation of Test 3: Hilden Figure 2a data with scores reversed to follow the normal convention'
        # scores (0, 1, 2, 3) high for negative were changed to (3, 2, 1, 0) respectively, high for positive
        scores = [ 3,  3,  3,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]  # 3b
        labels = [ 1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]  # 3b
        # groups = [[0.0, 0.5], [0.5, 1.0]]
        groups = [[0.0, 0.2], [0.2, 0.4], [0.4, 1.0]]

    elif testNum == 15:  # previously 3c
        descr = 'Test 15.(3c) A variation of Test 3: Hilden Figure 2a data with scores reversed to follow the normal convention'
        # scores (0, 1, 2, 3) high for negative were changed to (3, 2, 1, 0) respectively, high for positive
        scores = [ 3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]  # 3c
        labels = [ 1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]  # 3c
        #groups = [[0.0, 0.5], [0.5, 1.0]]
        groups = [[0.0, 0.2], [0.2, 0.4], [0.4, 1.0]]

    elif testNum == 16:  # previously 3d
        descr = 'Test 16.(3d) A variation of Test 3: Hilden Figure 2a data with scores reversed to follow the normal convention'
        scores = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1]  # 3d
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 3d
        groups = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 17:
        descr = 'Test 17. Hypothesis test re smaller group AUCnn_i'
        # creating ROC: (0,0), (0.33,0.6), (0.66,0.9), (1,1)
        # x ascends by thirds        - do 9  actual negatives
        # y ascends by 0.6, 0.3, 0.1 - do 10 actual positives
        # 3 sets of ties with scores: h=0.8, m=0.5, l=0.2
        h=0.8;  m=0.5;  l=0.2
        scores = [h, h, h, h, h, h, h, h, h, m, m, m, m, m, m, l, l, l, l].copy()
        labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0]
        groups = [[0.0, 1.0/3.0], [1.0/3.0, 2.0/3.0], [2.0/3.0, 1.0]]

    elif testNum == 18:
        descr = 'Test 18. Hypothesis test flipped re smaller group AUCnn_i'
        # similar to test 17 except labels and scores flipped
        # creating ROC: (0,0), (0.33,0.1), (0.66,0.4), (1,1)
        h=0.8;  m=0.5;  l=0.2
        scores = [l, l, l, l, l, l, l, l, l, m, m, m, m, m, m, h, h, h, h].copy()
        labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
        groups = [[0.0, 1.0/3.0], [1.0/3.0, 2.0/3.0], [2.0/3.0, 1.0]]

    elif testNum == 19:
        descr = 'Test 19. Testing'
        h = 0.8; m = 0.5; l = 0.2
        scores = [h, h, h, h, h, m, m, m, m, m, m, m, m, l, l, l ].copy()
        labels = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1 ]
        groups = [[0.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0]]

    elif testNum == 20:
        descr = 'Test 20. Example output from a decision tree'
        h = 1; l = 0
        scores = [h, h, h, h, h, l, l, l, l, l, l, l, l, l, l, l].copy()
        labels = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
        groups = [[0.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0]]

    else:
        if noError:
            scores, labels, groups, groupAxis, descr = [[], [], [], '', '']
        else:
            raise ValueError('Not a valid built-in test number.')
        #endif
    #endif
    return scores, labels, groups, groupAxis, descr
#enddef