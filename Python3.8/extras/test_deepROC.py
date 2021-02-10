# test_deepROC.py
#
# Copyright 2021 Ottawa Hospital Research Institute
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
#
# Revision history:
#   Original Matlab version by Andre Carrington, 2018
#   Ported to Python by Yusuf Sheikh and Andre Carrington, 2020
#   Imports the Transcript class
#   Additions and corrections from Partial-AUC-C to create DeepROC by Andre Carrington, 2021
#
# Functions:
#    makeLabels01
#    getSkewWithRates
#    getSkew
#    distance_point_to_line
#    optimal_ROC_point_indices
#    getListValuesByIndex
#    get_ROC_test_scores_labels_ranges
#    plotCalibrationCurve
#    calibrationOK
#    doCalibration
#    doAdvancedMeasures
#    test_deepROC
#
# inputs:  testNum, verbose
# outputs: results displayed and logged to a file
#
# Uses test vectors from Fawcett[1], Hilden[2], Carrington et al[3]
# This function runs tests that compute expected equalities for whole
# and partial AUC and c statistic measures discussed in Carrington
# et al [3]. It uses the trapezoid rule to compute area.
#
# Whole measures:     AUC,     c
# Partial measures:   cpAUC,   cDelta,  pAUC (pAUCy), pAUCx
# Normalized partial: cpAUCn,  cDeltan, pAUCn,        pAUCxn,  cLocal
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

# see beginning of runtime logic at the end of this file

# runtime parameters

global quiet, resultFile

quiet = False

# choose one of the following as input (file supercedes others if multiple):
useFile             = False
useSingleTestVector = True
useAllTestVectors   = False

# specify file input parameters
mydata              = ['040', '356', '529', '536', '581', '639', '643']
fileName            = f'input-matlab/result{mydata[2]}.mat'  # a matlab file (or future: csv file) for input
scoreVarColumn      = 'yscoreTest'                  # yscoreTest, yscore
targetVarColumn     = 'ytest'                       # ytest, yhatTest, yhat, ytrain
#scoreVarColumn      = 'yscore'                   # yscoreTest, yscore
#targetVarColumn     = 'yhat'                     # ytest, yhatTest, yhat, ytrain

# specify single test vector input parameters
singleTestVectorNum = 12  # which of 12 test vectors the function get_ROC_test_scores_labels_ranges below

# choose data science parameters
rangeAxis           = 'FPR'  # examine ranges (next) by FPR or TPR
filePAUCranges      = [[0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]  # ranges, as few or many as you like
useCloseRangePoint  = False  # automatically alter the ranges to match with the closest points in data
                             # useful if you want the discrete form of balanced accuracy to exactly match
                             # because we don't bother to interpolate that one
costs               = dict(cFP=1, cFN=1, cTP=0, cTN=0)  # specify relative costs explicitly (default shown)
#costs              = {}                                # use the default costs
rates               = False                             # treat costs as rates, e.g. cFPR (default False)

# choose what to show
sanityCheckWholeAUC = True
showPlot            = True
showData            = False
showError           = True     # areas in ROC plots that represent error

import do_pAUCc            as ac
import getPDF              as acKDE
import numpy               as np
import pandas              as pd
import scipy.io            as sio
import matplotlib.pyplot   as plt
import matplotlib.ticker   as ticker
import sklearn.calibration as cal
from   sklearn             import metrics
import time
import math
import transcript
from   os.path import splitext
import ntpath
import re
import dill
import pickle
import deepROC   as dr

# this function is a duplicate of the one in do_pAUCc.py
def makeLabels01(labels, posclass):
    ''' insert docstring here '''
    # STATUS: COMPLETE
    posIndex  = list(np.argwhere(np.array(labels) == posclass).flatten())
    negIndex  = list(np.argwhere(np.array(labels) != posclass).flatten())
    newlabels = labels.copy()
    for i in posIndex:
        newlabels[i] = 1
    for i in negIndex:
        newlabels[i] = 0
    return newlabels
#enddef

def getSkew_withRates(labels, posclass, rates=True, cFP=None, cFN=None, cTP=None, cTN=None):
    if rates is False:
        raise ValueError('rates should be True')
    cFPR, cFNR, cTPR, cTNR = (cFP, cFN, cTP, cTN)  # these are rates
    return getSkew(labels, posclass, rates=rates, cFP=cFPR, cFN=cFNR, cTP=cTPR, cTN=cTNR)
#enddef

def getSkew(labels, posclass, rates=None, cFP=None, cFN=None, cTP=None, cTN=None):
    labels  = makeLabels01(labels, posclass)  # forces  labels to the set {0,1}
    P       = int(sum(labels))                # assumes labels are in set {0,1}
    N       = len(labels) - P
    msg     = ''

    if rates == None or rates == False:
        # assumes costs are counts not rates
        ratetxt = ''
        ratechr = ''
    else:
        # but also handles costs as rates if needed
        ratetxt = 'rate unit '
        ratechr = 'R'
    # endif

    if cFP is None:
        msg = msg + f'\nCost of a false positive {ratetxt}(cFP{ratechr}): 1 (default, since not specified)'
        cFP = 1
    else:
        msg = msg + f'\nCost of a false positive {ratetxt}(cFP{ratechr}): {cFP:0.1f}'
    # endif

    if cFN is None:
        msg = msg + f'\nCost of a false negative {ratetxt}(cFN{ratechr}): 1 (default, since not specified)'
        cFN = 1
    else:
        msg = msg + f'\nCost of a false negative {ratetxt}(cFN{ratechr}): {cFN:0.1f}'
    # endif

    if cTP is None:
        msg = msg + f'\nCost of a true  positive {ratetxt}(cTP{ratechr}): 0 (default, since not specified)'
        cTP = 0
    else:
        msg = msg + f'\nCost of a true  positive {ratetxt}(cTP{ratechr}): {cTP:0.1f}'
    # endif

    if cTN is None:
        msg = msg + f'\nCost of a true  negative {ratetxt}(cTN{ratechr}): 0 (default, since not specified)'
        cTN = 0
    else:
        msg = msg + f'\nCost of a true  negative {ratetxt}(cTN{ratechr}): {cTN:0.1f}'
    # endif

    if len(msg):
        print(f'{msg}\n')
    #endif

    # assumes costs are counts (not rates)
    if rates == None or rates == False:
        skew = (N / P) * (cFP - cTN) / (cFN - cTP)
    else:
        cFPR, cFNR, cTPR, cTNR = (cFP, cFN, cTP, cTN)  # these are rates
        skew = ((N / P)**2) * (cFPR - cTNR) / (cFNR - cTPR)
        # see paper by Carrington for derivation
    return skew, N, P
#enddef

def distance_point_to_line(qx, qy, px, py, m):
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

def optimal_ROC_point_indices(fpr, tpr, skew):
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

def getListValuesByIndex(a, idx_list):
    b = []
    for i in idx_list:
        b = b + [a[i]]
    return b
# enddef

def get_ROC_test_scores_labels_ranges(testNum):
    if testNum == 1:  # old testNum 1
        descr  = 'Test 1. Fawcett Figure 3 data (balanced classes) with partial curve boundaries ' + \
                 'aligned with instances on step verticals'
        scores = np.array([ 0.9,  0.8,  0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38,
                           0.37, 0.36, 0.35, 0.34, 0.33, 0.30, 0.1])
        labels = np.array([   1,    1,    0,   1,    1,    1,    0,    0,    1,     0,   1,    0,    1,
                              0,    0,    0,   1,    0,    1,    0])
        pAUCranges = [[0.0, 0.3], [0.3, 0.5], [0.5, 1.0]]

    elif testNum == 2:  # old testNum 2 (scores made proper for reference, no effect on measurement)
        descr  = 'Test 2. Carrington Figure 7 data (with a 1:3 P:N class imbalance) with partial curve ' + \
                 'boundaries aligned with instances.'
        # This has the same scores as Carrington Figure 8, and scores similar to Fawcett Figure 3,
        # but the labels are altered for class imbalance
        scores = [ 0.95,  0.9,  0.8,  0.7, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.5, 0.49, 0.48, 0.47,
                   0.46, 0.45, 0.44, 0.43, 0.40, 0.2]
        labels = [    1,    1,    0,    1,    1,    0,    0,    0,    0,    0,   0,    0,    1,    0,
                      0,    0,    0,    0,    0,    0]
        pAUCranges = [[0.0, 0.2], [0.2, 0.4], [0.4, 1.0]]

    elif testNum == 3:  # no old testNum equivalent
        descr  = 'Test 3. Hilden Figure 2a data with scores reversed to follow the normal convention'
        # scores (0, 1, 2, 3) high for negative were changed to (3, 2, 1, 0) respectively, high for positive
        scores = [ 3,  3,  3,  3,  3,  3,  3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
        labels = [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        pAUCranges = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 4:  # no old testNum equivalent
        descr = 'Test 4. Hilden Figure 2b data with scores reversed to follow the normal convention'
        # scores (0, 1, 2, 3) high for negative were changed to (3, 2, 1, 0) respectively, high for positive
        scores = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0]
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pAUCranges = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 5:  # old testNum 5 (scores made proper for reference, no effect on measurement)
        descr  = 'Test 5. Hilden Figure 2c data with scores reversed to follow the normal convention'
        # scores (0, 1, 2, 3) high for negative were changed to (3, 2, 1, 0) respectively, high for positive
        # histogram counts were divided by 5, so that (50, 35, 15) became (10, 7, 3)
        scores = [   0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
        labels = [   0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        pAUCranges = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 6:  # old testNum 6
        descr  = 'Test 6. Fawcett Figure 3 data with partial curve boundaries aligned with instances ' + \
                 'on step horizontals'
        scores = [ 0.9,  0.8,  0.7,  0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37,
                  0.36, 0.35, 0.34, 0.33, 0.30, 0.1 ]
        labels = [   1,    1,    0,    1,    1,   1,    0,    0,    1,     0,   1,    0,    1,    0,
                     0,    0,    1,    0,    1,   0 ]
        pAUCranges = [[0.0, 0.2], [0.2, 0.6], [0.6, 1.0]]

    elif testNum == 7:  # old testNum 7
        descr  = 'Test 7. Fawcett Figure 3 data with partial curve boundaries not aligned with '\
                 'instances, requiring interpolation'
        scores = [ 0.9,  0.8,  0.7,  0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37,
                  0.36, 0.35, 0.34, 0.33, 0.30, 0.1 ]
        labels = [   1,    1,    0,    1,    1,    1,    0,    0,    1,     0,   1,    0,    1,    0,
                     0,    0,    1,    0,    1,    0]
        pAUCranges = [[0.0, 0.17], [0.17, 0.52], [0.52, 1.0]]

    elif testNum == 8:  # old testNum 8
        descr  = 'Test 8. Carrington Figure 4 data with the same shape and measure as Fawcett Figure 3 ' + \
                 'but with different scores'
        scores = [ 0.95,  0.9,  0.8,  0.7, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.5, 0.49, 0.48, 0.47,
                   0.46, 0.45, 0.44, 0.43, 0.40, 0.2]
        labels = [    1,    1,    0,    1,    1,    1,    0,    0,    1,    0,   1,    0,    1,    0,
                      0,    0,    1,    0,    1,   0]
        pAUCranges = [[0.0, 0.17], [0.17, 0.52], [0.52, 1.0]]
    #endif

    elif testNum == 9:  # old testNum 9
        descr  = 'Test 9. Carrington Figure 7 data: same instances as Carrington Figure 4, but different ' + \
                 'labels'
        scores = [ 0.95,  0.9,  0.8,  0.7, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.5, 0.49, 0.48, 0.47,
                   0.46, 0.45, 0.44, 0.43, 0.40, 0.2]
        labels = [    1,    1,    0,    1,    1,    0,    0,    0,    0,    0,   0,    0,    1,    0,
                      0,    0,    0,    0,    0,    0]
        pAUCranges = [[0.0, 0.17], [0.17, 0.52], [0.52, 1.0]]

    elif testNum == 10:  # old testNum 3
        descr  = 'Test 10. Carrington Figure 8 data'
        scores = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pAUCranges = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 11:  # old testNum 4
        descr  = 'Test 11. A classifier that does worse than "continuous" chance'
        scores = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        labels = [  0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
        pAUCranges = [[0.0, 0.2], [0.2, 0.5], [0.5, 1.0]]

    elif testNum == 12:
        descr = 'Test 12. Simplest test'
        scores = [0.8, 0.7, 0.7, 0.6]
        labels = [  1,   1,   0,   0]
        pAUCranges = [[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]]

    else:
        raise ValueError('Not a valid built-in test number.')
    return scores, labels, pAUCranges, descr
#enddef

def plotCalibrationCurve(plotTitle, dataTitle, params):
    prob_true, prob_predicted = \
        cal.calibration_curve(**params)
        #cal.calibration_curve(labels, scores, normalize=False, strategy=strategy, n_bins=bins)
    actual_bins = len(prob_true)
    bins = params['n_bins']
    if bins > actual_bins:
        print(f'Used {actual_bins} bins instead of the {bins} bins requested.')
    #endif
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(prob_predicted, prob_true, "s-", label=dataTitle)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel('Predicted risk/probability')
    plt.ylabel('Observed risk/probability')
    plt.title(plotTitle)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    ax.axes.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.axes.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    plotGrey = lambda x, y: plt.fill(x, y, 'k', alpha=0.3, linewidth=None)
    x = []
    y = []
    shadeWidth = int(round(actual_bins / 3))
    step       = shadeWidth * 2     # 3 bins=skip 2;  6 bins=skip 4;  9 bins=skip 6
    for i in range(0, actual_bins, step):
        x0 = i     * 1/actual_bins
        x1 = (i+shadeWidth) * 1/actual_bins
        x = x + [x0] + [x0] + [x1] + [x1]
        y = y +  [0] +  [1] +  [1] +  [0]
    #endfor
    plotGrey(x, y)

    return fig, ax
# enddef

def calibrationOK(numScores, bins):
    if   (numScores/25) >= bins:
        return 1
    elif (numScores / 10) >= bins:
        return 0.5
    else:
        return 0
#endif

def doCalibration(scores, labels, posclass, fileNum):

    scores, newlabels, labels = ac.sortScoresFixLabels(scores, labels, posclass, True) # True = ascending

    maxScore  = float(max(scores))
    minScore  = float(min(scores))
    print(f'Before: min,max = {minScore},{maxScore}')
    scores_np = (np.array(scores) - minScore) / (maxScore - minScore)
    maxScore  = float(max(scores_np))
    minScore  = float(min(scores_np))
    print(f'After: min,max = {minScore},{maxScore}')
    numScores  = int(len(scores_np))

    quiet = True
    Xc_cts, Y = acKDE.getPDF(scores_np, 'epanechnikov', 'new', quiet)
    Y1D = Y[:, 0]
    y2 = np.interp(scores_np, Xc_cts, Y1D)
    plt.plot(scores_np, y2)
    plt.show()

    for bins in [3, 6, 9]:
        if calibrationOK(numScores, bins) == 0:
            print(f'Not plotted: insufficient scores ({numScores}) for {bins} bins')
        else:
            plotTitle = f'Calibration plot with {bins} bins'
            dataTitle = 'Classifier'
            if calibrationOK(numScores, bins) == 0.5:
                print(f'Plotted despite insufficient scores ({numScores}) for {bins} bins')
            # endif
            params = dict(y_true=newlabels, y_prob=scores_np, normalize=False, strategy='uniform', n_bins=bins)
            fig, ax = plotCalibrationCurve(plotTitle, dataTitle, params)
        # endif
        plt.show()
        fig.savefig(f'output/calib_{fileNum}-{bins}.png')
    # endfor
#enddef


def doAdvancedMeasures(scores, labels, groupAxis, deepROC_groups, testNum):
    global quiet
    costs = dict(cFP=1, cFN=1, cTP=0, cTN=0, rates=False)
    results, EQresults = dr.deepROC(costs=costs,     showPlot=True,  showData=False,
                                    showError=False, scores=scores,   labels=labels,  posclass=1,
                                    testNum=testNum, pAUCranges=deepROC_groups,   rangeAxis=groupAxis,
                                    useCloseRangePoint=False, sanityCheckWholeAUC=True, quiet=quiet)
    return results, EQresults
#enddef

def test_deepROC(testNum=1, costs={}, sanityCheckWholeAUC=True, showPlot=True, showData=True,
                 showError=True, useCloseRangePoint=False, fileName='', filePAUCranges={},
                 scoreNameNum=0, labelNameNum=1, rangeAxis='FPR'):

    global quiet, resultFile

    # choose the fileNum to use in the logfile
    if fileName == '':
        fileNum = testNum
    else:  # reduce filename to any 3-digit log number it contains, if possible
        fileNameBase = ntpath.basename(fileName)
        fileNameBase = splitext(fileNameBase)[0]  # remove the extension
        match        = re.search(r'\d\d\d', fileNameBase)
        if match:
            fileNum = match.group()
        else:
            fileNum = fileNameBase
        #endif
    #endif

    # capture standard out to logfile
    if showData:
        logfn = 'output/log_test_case' + str(fileNum) + '_verbose.txt'
    else:
        logfn = 'output/log_test_case' + str(fileNum) + '.txt'
    #endif
    transcript.start(logfn)
    resultFile = open(f'output/results_case{testNum}.pkl', 'wb')

    print('Output text file: ', logfn)
    if fileName == '':
        print(f'Test data: {fileNum} (built-in)')
    else:
        print(f'Test data: {fileNum} (from {fileName})')
    #endif
    tic         = time.perf_counter()

    posclass    = 1   # numeric scalar

    if fileName == '':
        scores, labels, pAUCranges, descr = get_ROC_test_scores_labels_ranges(testNum)
        print(f'Description: {descr}')
    else:
        if fileName[-4:] == '.mat':  # if matlab file input
            try:
                fileContent = sio.loadmat(fileName)  # handle any file not found errors naturally
                scores      = fileContent[scoreNameNum]
                labels      = fileContent[labelNameNum]
            except:
                raise ValueError(f'File {fileName} is either not found or is not a matlab file')
            #endtry
        else:  # otherwise assume a CSV file input
            try:
                file_df = pd.read_csv(fileName)
                scores  = file_df[scoreNameNum]
                labels  = file_df[labelNameNum]
            except:
                raise ValueError(f'File {fileName} is either not found or is not a CSV file')
            #endtry
        #endif

        if len(filePAUCranges) > 0:
            pAUCranges = filePAUCranges
        else:
            pAUCranges = [[0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]
        #endif
    #endif

    deepROC_groups = [[0.0, 0.5], [0.5, 1.0]]
    #deepROC_groups = [[0.0, 0.167], [0.167, 0.333], [0.333, 0.5],
    #                  [0.5, 0.667], [0.667, 0.833], [0.833, 1.0]]  # note: whole ROC is
                                                                    # automatically included
                                                                    # as group 0
    groupAxis = 'FPR'  # FPR, TPR or any of: Predicted Risk, Probability or Score
    areaMeasures = ['AUC', 'AUC_full', 'AUC_plain', 'AUC_micro', 'AUPRC']
    groupMeasures = ['cDelta', 'cpAUC', 'pAUC', 'pAUCx',
                     'cDeltan', 'cpAUCn', 'pAUCn', 'pAUCxn',
                     'avgA', 'bAvgA', 'avgSens', 'avgSpec',
                     'avgPPV', 'avgNPV',
                     'avgLRp', 'avgLRn',
                     'ubAvgA', 'avgBA', 'sPA']
    num_group_measures = len(groupMeasures)
    num_area_measures = len(areaMeasures)

    total_folds = 1
    MAX_TRIALS = 1
    num_groups = len(deepROC_groups) + 1  # automatically include (add) whole ROC as group 0
    areaMatrix = np.zeros(shape=[total_folds, num_area_measures, MAX_TRIALS])
    #                              30           x5                 x100
    groupMatrix = np.zeros(shape=[total_folds, num_groups, num_group_measures, MAX_TRIALS])
    #                              30           x6          x15                 x100 = 270k * 4B = 1.08 MB

    logTestNum = f'case{testNum}'
    results, EQresults = doAdvancedMeasures(scores, labels, groupAxis, deepROC_groups, logTestNum)

    fold      = 0
    trial_num = 0
    for group in range(0, num_groups):
        # note: these are validation results, not training results
        if group == 0:
            areaMatrix[fold, :, trial_num]     = np.array([results[group][m] for m in areaMeasures])
        # endif
        groupMatrix[fold, group, :, trial_num] = np.array([results[group][m] for m in groupMeasures])
    # endfor

    # mean measure is across folds, but only 1 fold here
    measure_to_optimize = 'AUC'
    type_to_optimize    = 'area'
    auc_index           = 0  # AUC
    mean_measure        = areaMatrix[fold, auc_index, trial_num]

    # Store settings
    pickle.dump([measure_to_optimize, type_to_optimize, deepROC_groups,
                 groupAxis, areaMeasures, groupMeasures], resultFile, pickle.HIGHEST_PROTOCOL)
    # Store performance
    pickle.dump([mean_measure, areaMatrix, groupMatrix], resultFile, pickle.HIGHEST_PROTOCOL)

    transcript.stop()
    return EQresults
#enddef

if   useFile == True:
    passTest = test_deepROC(testNum=singleTestVectorNum, costs=costs,
                          sanityCheckWholeAUC=sanityCheckWholeAUC,
                          showPlot=showPlot, showData=showData, showError=showError,
                          useCloseRangePoint=useCloseRangePoint,
                          fileName=fileName, filePAUCranges=filePAUCranges,
                          scoreNameNum=scoreVarColumn, labelNameNum=targetVarColumn,
                          rangeAxis=rangeAxis)
elif useSingleTestVector == True:
        passTest = test_deepROC(testNum=singleTestVectorNum, costs=costs,
                          sanityCheckWholeAUC=sanityCheckWholeAUC,
                          showPlot=showPlot, showData=showData, showError=showError,
                          useCloseRangePoint=useCloseRangePoint)
elif useAllTestVectors == True:
    passAllTests = True
    numAllTests  = 11
    passTest     = np.zeros((numAllTests,))
    for testNum in range(1, numAllTests+1):
        passTest[int(testNum-1)] = \
            test_deepROC(testNum=testNum, costs=costs,
                       sanityCheckWholeAUC=sanityCheckWholeAUC,
                       showPlot=showPlot, showData=showData, showError=showError,
                       useCloseRangePoint = useCloseRangePoint)
        passAllTests = passAllTests and passTest[int(testNum-1)]
    #endfor
    i = 1
    for result in passTest:
        if result == True:
            print(f"Test {i}: All results matched (passed).")
        else:
            print(f"Test {i}: Some results did not match (some failed).")
        #endif
        i = i + 1
    #endfor
    if passAllTests == True:
        print(f"All results in all tests matched (passed).")
    else:
        print(f"Some results in tests did not match (some failed).")
    #endif
#endif
