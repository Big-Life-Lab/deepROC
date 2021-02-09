# test_pAUCc.py
#
# Copyright 2020 Ottawa Hospital Research Institute
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
#
# Functions:
#    makeLabels01
#    getSkewWithRates
#    getSkew
#    distance_point_to_line
#    optimal_ROC_point_indices
#    getListValuesByIndex
#    get_ROC_test_scores_labels_ranges
#    test_pAUCc
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
# Partial measures:   pAUCc,   cDelta,  pAUC (pAUCy), pAUCx
# Normalized partial: pAUCcn,  cDeltan, pAUCn,        pAUCxn,  cLocal
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

# choose one of the following as input (file supercedes others if multiple):
useFile             = True
useSingleTestVector = False
useAllTestVectors   = False

# specify corresponding input parameters
fileName            = 'input-matlab/result581.mat'  # a matlab file (or future: csv file) for input
singleTestVectorNum = 1  # which of 11 test vectors from the function get_ROC_test_scores_labels_ranges below

# choose data science parameters
rangeAxis           = 'FPR'  # examine ranges (next) by FPR or TPR
filePAUCranges      = [[0, 0.33], [0.33, 0.67], [0.67, 1.0]]  # ranges, as few or many as you like
useCloseRangePoint  = True   # automatically alter the ranges to match with the closest points in data
costs               = dict(cFP=1, cFN=1, cTP=0, cTN=0)  # specify relative costs explicitly (default shown)
#costs              = {}                                # use the default costs
rates               = False                             # treat costs as rates, e.g. cFPR (default False)

# choose what to show
sanityCheckWholeAUC = True
showPlot            = True
showData            = False
showError           = True

import do_pAUCc          as ac
import numpy             as np
import scipy.io          as sio
from   sklearn           import metrics
import time
import math
import transcript
from   os.path import splitext
import ntpath
import re

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
    return skew
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

    else:
        raise ValueError('Not a valid built-in test number.')
    return scores, labels, pAUCranges, descr
#enddef

def test_pAUCc(testNum=1, costs={}, sanityCheckWholeAUC=True, showPlot=True, showData=True,
               showError=True, fileName='', filePAUCranges={}, rangeAxis='FPR', useCloseRangePoint=False):

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
        logfn   = 'output/log_pAUCc_case' + str(fileNum) + '_verbose.txt'
    else:
        logfn = 'output/log_pAUCc_case' + str(fileNum) + '.txt'
    #endif
    transcript.start(logfn)

    print('Output text file: ', logfn)
    if fileName == '':
        print(f'Test data: {fileNum} (built-in)')
    else:
        print(f'Test data: {fileNum} (from {fileName})')
    #endif
    tic         = time.perf_counter()

    if fileName == '':
        scores, labels, pAUCranges, descr = get_ROC_test_scores_labels_ranges(testNum)
        print(f'Description: {descr}')
    else:
        try:
            fileContent = sio.loadmat(fileName)  # handle any file not found errors naturally
        except:
            raise ValueError(f'File {fileName} is either not found or is not a matlab file')
        #endtry
        scores = fileContent['yscoreTest']
        labels = fileContent['ytest']
        #scores = fileContent['yscore']
        #labels = fileContent['yhatTest']
        #labels = fileContent['yhat']
        if len(filePAUCranges) > 0:
            pAUCranges = filePAUCranges
        else:
            pAUCranges = [[0.0, 0.33], [0.33, 0.66], [0.66, 1.0]]
        #endif
    #endif
    # I have seen round-off errors in the range of approx 1.0e-16
    print('\nThere are sometimes numerical differences due to round-off error or other \n'
          'computational artifacts. We therefore test for equality within epsilon=1.0e-12.')
    ep = 1*(10**-12)

    showThresh  = 25  # show 20 scores per plot
    posclass    = 1   # numeric scalar

    # Get standard fpr, tpr, thresholds from the labels scores
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=posclass)
    # data are returned, sorted in ascending order by fpr, then tpr within fpr

    # sklearn tries to offer a full roc but it is insufficient:
    #    - it only shows hidden/intermediate points with unique thresholds
    #    - i.e., it does not show ties (e.g., try test vector #3)
    #    - it inserts a fake threshold of max+1 at the (0,0) point, when it really should be inf or anything > max
    #    - it sets the threshold at (1,1) to the min value, when it really should be -inf or anything <= min
    showSklearnFullROC = False
    if showSklearnFullROC:
        skfpr, sktpr, skthresh = metrics.roc_curve(labels, scores, pos_label=posclass, drop_intermediate=False)
        print('Python\'s built-in full ROC data:')
        print('%-6s, %-6s, %-6s' % ('fpr', 'tpr', 'thresh'))
        for x, y, t in zip(skfpr, sktpr, skthresh):
            print(f'{x:-6.3g}, {y:-6.3g}, {t:-6.3g}')
        # endfor
        ac.plotSimpleROC(skfpr, sktpr, 'Sklearn Full ROC')
    # endif

    AUC = metrics.auc(fpr, tpr)
    print(f"{'Python AUC':15s} = {AUC:0.4f}")

    if ('rates' in costs) and costs['rates'] is True:
        skew = getSkew_withRates(labels, posclass, rates, **costs)
        print(f'Skew (slope) = N * N * cFPR-cTNR = {skew:0.1f}')
        print( '               -   -   ---------')
        print( '               P   P   cFNR-cTPR')
    else:
        skew = getSkew(labels, posclass, rates, **costs)
        print(f'Skew (slope) = N * cFP-cTN = {skew:0.1f}')
        print( '               -   -------')
        print( '               P   cFN-cTP')
    #endif
    print(' ')

    # Get and show the optimal ROC points (per Metz)
    opt_idx      = optimal_ROC_point_indices(fpr, tpr, skew)
    fpr_opt      = getListValuesByIndex(fpr, opt_idx)
    tpr_opt      = getListValuesByIndex(tpr, opt_idx)
    thresh_opt   = getListValuesByIndex(thresholds, opt_idx)
    for fprx, tprx, threshx in zip(fpr_opt, tpr_opt, thresh_opt):
        print(f"{'optimalROCpoint':15s} = ({fprx:0.4f},{tprx:0.4f}) at threshold: {threshx:0.4f}")
    #endfor

    if sanityCheckWholeAUC == True:
        pAUCranges.insert(0, [0, 1])
        indices  = np.arange(0, len(pAUCranges))    # include index 0 as wholeAUC
    else:
        indices  = np.arange(1, len(pAUCranges)+1)  # index 1,2,... for partial curves
    #endif

    passALL = True; cDelta_sum = 0; pAUC_sum  = 0; pAUCx_sum  = 0; pAUCc_sum = 0
    for pAUCrange, index in zip(pAUCranges, indices):
        passEQ, cDelta, pAUCc, pAUC, pAUCx, cDeltan, pAUCcn, pAUCn, pAUCxn, extras_dict \
            = ac.do_pAUCc(mode='testing',           index=index,         pAUCrange=pAUCrange,
                          labels=labels,            scores=scores,       posclass=posclass,
                          fpr=fpr,                  tpr=tpr,             thresh=thresholds,
                          fpr_opt=fpr_opt,          tpr_opt=tpr_opt,     thresh_opt=thresh_opt,
                          numShowThresh=showThresh, testNum=fileNum,     showPlot=showPlot,
                          showData=showData,        showError=showError, ep=ep,
                          rangeAxis=rangeAxis,      useCloseRangePoint=useCloseRangePoint)
        passALL = passALL and passEQ
        if index > 0:
            cDelta_sum = cDelta_sum + cDelta
            pAUCc_sum  = pAUCc_sum  + pAUCc
            pAUC_sum   = pAUC_sum   + pAUC
            pAUCx_sum  = pAUCx_sum  + pAUCx
        #endif
        if index == 0:
            c = extras_dict['c']
        #endif
    #endfor

    # code to check for PASS here
    pass1 = ac.areEpsilonEqual(cDelta_sum, c,   'cDelta_sum', 'c',   ep)
    pass2 = ac.areEpsilonEqual(pAUCc_sum,  AUC, 'pAUCc_sum',  'AUC', ep)
    pass3 = ac.areEpsilonEqual(pAUC_sum ,  AUC, 'pAUC_sum',   'AUC', ep)
    pass4 = ac.areEpsilonEqual(pAUCx_sum,  AUC, 'pAUCx_sum',  'AUC', ep)
    passALL = passALL and pass1 and pass2 and pass3 and pass4

    toc = time.perf_counter()
    print(f"Analysis performed and plotted in {toc - tic:0.1f} seconds.")
    transcript.stop()
    return passALL
#enddef

if   useFile == True:
    passTest = test_pAUCc(testNum=singleTestVectorNum, costs=costs,
                          sanityCheckWholeAUC=sanityCheckWholeAUC,
                          showPlot=showPlot, showData=showData, showError=showError,
                          fileName=fileName, filePAUCranges=filePAUCranges,
                          rangeAxis=rangeAxis, useCloseRangePoint=useCloseRangePoint)
elif useSingleTestVector == True:
        passTest = test_pAUCc(testNum=singleTestVectorNum, costs=costs,
                          sanityCheckWholeAUC=sanityCheckWholeAUC,
                          showPlot=showPlot, showData=showData, showError=showError)
elif useAllTestVectors == True:
    passAllTests = True
    numAllTests  = 11
    passTest     = np.zeros((numAllTests,))
    for testNum in range(1, numAllTests+1):
        passTest[int(testNum-1)] = \
            test_pAUCc(testNum=testNum, costs=costs,
                       sanityCheckWholeAUC=sanityCheckWholeAUC,
                       showPlot=showPlot, showData=showData, showError=showError)
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