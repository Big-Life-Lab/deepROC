#!/usr/bin/env python
# -*- coding: latin-1 -*-
# ROCFunctions.py
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

def Heaviside(x):  # handles ties properly
    if x > 0:
        return 1
    elif x < 0:
        return 0
    elif x == 0:
        return 0.5
    else:
        raise ValueError('Unexpected value for x')
    # endif
# enddef

def C_statistic(predicted_scores, labels):
    '''Computes and returns the C statistic (a discrete measure)'''
    if predicted_scores is None or labels is None:
        SystemError('Actual labels and predicted scores are required to compute the C statistic.')

    labels, poslabel     = checkFixLabels(labels, poslabel=None)  # if labels not {0,1} nor {True,False}
                                                                  # then sets {0,1}
    posScores, negScores = get_positive_negative_scores(predicted_scores, labels)

    P = len(posScores)
    N = len(negScores)
    U = 0  # C is a U statistic (with tie handling) prior to normalization
    # the U statistic defined in literature and commonly used does not handle ties properly

    for j in range(0, P):
        for k in range(0, N):
            U = U + Heaviside(posScores[j] - negScores[k])  # handles ties properly
        # endfor
    # endfor
    C = U / (P * N)  # normalization
    return C
# enddef

def makeLabels01(labels, poslabel):
    import numpy as np
    posIndex = list(np.argwhere(np.array(labels) == poslabel).flatten())
    negIndex = list(np.argwhere(np.array(labels) != poslabel).flatten())
    newlabels = labels.copy()
    for i in negIndex:
        newlabels[i] = 0
    for i in posIndex:
        newlabels[i] = 1
    return newlabels
# enddef

def checkFixLabels(labels, poslabel=None):
    import numpy as np

    unique_labels = np.unique(labels)
    if len(unique_labels) > 2:
        SystemError('More than 2 unique labels')

    # higherlabel used if poslabel not specified
    if unique_labels[0] > unique_labels[1]:
        higherlabel = unique_labels[0]
        lowerlabel  = unique_labels[1]
    else:
        higherlabel = unique_labels[1]
        lowerlabel  = unique_labels[0]
    #endif

    if (higherlabel == 1 or higherlabel == True) and \
       (lowerlabel  == 0 or lowerlabel  == False):
        return labels, higherlabel
    #endif

    if poslabel is None:
        # if poslabel unspecified then use the higherlabel as positive
        newlabels = makeLabels01(labels, higherlabel)
    else:
        newlabels = makeLabels01(labels, poslabel)
    #endif
    newposlabel = 1

    return newlabels, newposlabel
#enddef

def sortScoresFixLabels(scores, labels, posclass, ascending):
    import numpy as np

    # use newlabels to make sure the positive class has a higher label value
    newlabels, newposclass = checkFixLabels(labels, poslabel=posclass)

    # sort in descending order by scores, then by newlabels (within tied scores)
    # (we must do this ascending, and then reverse it)
    dtype     = [('scores', float), ('newlabels', int), ('labels', int)]  # assumes labels are int
    rocTuples = list(zip(scores, newlabels, labels))
    rocArray  = np.array(rocTuples, dtype=dtype)
    temp      = np.sort(rocArray,   order=['scores', 'newlabels'])
    if ascending:
        scores, newlabels, labels = zip(*temp)
    else:
        final = temp[::-1]        # reverse it
        # put the sorted data back into the original list variables
        scores, newlabels, labels = zip(*final)
    #endif
    return scores, newlabels, labels
#enddef

def get_positive_negative_scores(scores, labels):
    # takes scores, labels as lists
    # returns posScores, negScores as lists
    import numpy as np

    labels_nd  = np.array(labels)
    posIndex   = list(np.argwhere(labels_nd == 1).flatten())
    negIndex   = list(np.argwhere(labels_nd == 0).flatten())
    scores_nd  = np.array(scores)
    posScores  = list(scores_nd[posIndex])
    negScores  = list(scores_nd[negIndex])
    return posScores, negScores
#enddef

def getSlopeOrSkew(NPclassRatio, costs, quiet=True):

    msg     = ''
    costsAreRates = costs['costsAreRates']

    if not costsAreRates:
        cFP, cFN, cTP, cTN = costs['cFP'], costs['cFN'], costs['cTP'], costs['cTN']

        if cFP is None:
            msg = msg + f'\nCost of a false positive (cFP): 1 (default, since unspecified)'
            cFP = 1
        else:
            msg = msg + f'\nCost of a false positive (cFP): {cFP:0.1f}'
        #endif

        if cFN is None:
            msg = msg + f'\nCost of a false negative (cFN): 1 (default, since unspecified)'
            cFN = 1
        else:
            msg = msg + f'\nCost of a false negative (cFN): {cFN:0.1f}'
        # endif

        if cTN is None:
            msg = msg + f'\nCost of a true negative (cTN): 0 (default, since unspecified)'
            cFN = 0
        else:
            msg = msg + f'\nCost of a true negative (cTN): {cTN:0.1f}'
        # endif

        if cTP is None:
            msg = msg + f'\nCost of a true positive (cTP): 0 (default, since unspecified)'
            cTP = 0
        else:
            msg = msg + f'\nCost of a true positive (cTP): {cTP:0.1f}'
        # endif

        if not quiet:
            print(f'{msg}\n')
        skew = NPclassRatio * (cFP - cTN) / (cFN - cTP)

    else:
        cFPR, cFNR, cTPR, cTNR = costs['cFP'], costs['cFN'], costs['cTP'], costs['cTN']
        # cFPR, cFNR, cTPR, cTNR = costs['cFPR'], costs['cFNR'], costs['cTPR'], costs['cTNR']

        if cFPR is None:
            msg = msg + f'\nCost of a false positive rate unit (cFPR): 1 (default, since unspecified)'
            cFPR = 1
        else:
            msg = msg + f'\nCost of a false positive rate unit (cFPR): {cFPR:0.1f}'
        #endif
        if cFNR is None:
            msg = msg + f'\nCost of a false negative rate unit (cFNR): 1 (default, since unspecified)'
            cFNR = 1
        else:
            msg = msg + f'\nCost of a false negative rate unit (cFNR): {cFNR:0.1f}'
        #endif
        if cTPR is None:
            msg = msg + f'\nCost of a true positive rate unit (cTPR): 1 (default, since unspecified)'
            cTPR = 1
        else:
            msg = msg + f'\nCost of a true positive rate unit (cTPR): {cTPR:0.1f}'
        #endif
        if cTNR is None:
            msg = msg + f'\nCost of a true negative rate unit (cTNR): 1 (default, since unspecified)'
            cTNR = 1
        else:
            msg = msg + f'\nCost of a true negative rate unit (cTNR): {cTNR:0.1f}'
        #endif

        if not quiet:
            print(f'{msg}\n')
        skew = (NPclassRatio ** 2) * (cFPR - cTNR) / (cFNR - cTPR)
    #endif

    return skew
#enddef

def distance_point_to_line(qx, qy, px, py, m):
    import math
    import numpy as np

    # point p (px,py) and slope m define a line
    # query point q (qx,qy): how far is q from the line?
    if m == 0:
        # slope is zero (horizontal line), distance is vertical only
        return abs(qy-py)
        #raise ValueError('slope is zero')
    #endif
    if np.isinf(m):
        # slope is infinite (vertical line), distance is horizontal only
        return abs(qx-px)
        #raise ValueError('slope is infinite')
    #endif

    # line through p:            y =     m *(x-px)+py
    # perpendicular slope:            -1/m
    # perpendicular line from q: y = (-1/m)*(x-qx)+qy
    # equate both lines to find intersection at (x0,y0)
    x0 = (m*px - py + qx/m + qy) / (m + 1/m)
    # then find y_0 using first line definition
    y0 = m*(x0-px)  + py
    return math.sqrt( (qx-x0)**2 + (qy-y0)**2 )
#enddef

def plot_major_diagonal():
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(0, 1, 3)
    plt.plot(x, x, linestyle=':', color='black')  # default linewidth is 1.5
    plt.plot(x, x, linestyle='-', color='black', linewidth=0.25)
#enddef

def optimal_ROC_point_indices(fpr, tpr, skew):
    import math
    import numpy as np

    n       = len(fpr)
    mindist = math.inf
    min_idx = []
    # define a line with slope skew passing through the top-left ROC point (x,y)=(0,1)
    # for each point in (fpr,tpr) get the distance from that point to the line
    for i in np.arange(0, n):
        d = distance_point_to_line(fpr[i], tpr[i], 0, 1, skew)
        #print(f'd: {d}')
        if   d == mindist:
            min_idx = min_idx + [i]
        elif d < mindist:
            mindist = d
            min_idx = [i]
        #endif
    #endif
    # now min_idx contains the indices for the point(s) with the minimum distance
    # return those indices for the optimal ROC points
    return min_idx
#enddef

def sortScoresAndLabels2(scores, newlabels):
    # 2 refers to sorting a 2-variable tuple
    import numpy as np
    # sort in descending order by scores, then by newlabels (within tied scores)
    # (we must do this ascending, and then reverse it)
    # the following assumes labels are int
    dtype2     = [('scores', float), ('newlabels', int)]
    rocTuples2 = list(zip(scores, newlabels))
    rocArray2  = np.array(rocTuples2, dtype=dtype2)
    temp2      = np.sort(rocArray2, order=['scores', 'newlabels'])
    final2     = temp2[::-1]  # reverse the order of elements
    # put the sorted data back into the original list variables
    scores, newlabels = zip(*final2)  # zip returns (immutable) tuples
    # we could convert these to lists, but we don't need to: scores = list(scores)
    return scores, newlabels
#enddef

def range01Check(range):
    if range[0] < 0 or range[0] > 1 \
            or range[1] < 0 or range[1] > 1:
        raise ValueError('Improper range: it must be in [0, 1]')
    return
#enddef