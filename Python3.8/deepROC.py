# deepROC.py
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
#    computeCalibrationCurve
#    plotCalibrationCurve
#    calibrationOK
#    doCalibration
#    doAdvancedMeasures
#    deepROC
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

def getSkew_withRates(labels, posclass, quiet, rates=True, cFP=None, cFN=None, cTP=None, cTN=None):
    if rates is False:
        raise ValueError('rates should be True')
    cFPR, cFNR, cTPR, cTNR = (cFP, cFN, cTP, cTN)  # these are rates
    return getSkew(labels, posclass, quiet, rates=rates, cFP=cFPR, cFN=cFNR, cTP=cTPR, cTN=cTNR)
#enddef

def getSkew(labels, posclass, quiet, rates=None, cFP=None, cFN=None, cTP=None, cTN=None):
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

    if len(msg) and not quiet:
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

def computeCalibrationCurve(plotTitle, dataTitle, quiet, params):
    prob_true, prob_predicted =  cal.calibration_curve(**params)
    # cal.calibration_curve(labels, scores, normalize=False, strategy=strategy, n_bins=bins)
    actual_bins = len(prob_true)
    bins = params['n_bins']
    if bins > actual_bins and not quiet:
        print(f'Used {actual_bins} bins instead of the {bins} bins requested.')
    #endif
    # plt.plot(prob_predicted, prob_true, "s-", label=dataTitle)
    # plt.xlabel('Predicted risk/probability')
    # plt.ylabel('Observed risk/probability')
    return prob_predicted, prob_true
# enddef

def plotCalibrationCurve(plotTitle, dataTitle, quiet, params):
    prob_true, prob_predicted = \
        cal.calibration_curve(**params)
        #cal.calibration_curve(labels, scores, normalize=False, strategy=strategy, n_bins=bins)
    actual_bins = len(prob_true)
    bins = params['n_bins']
    if bins > actual_bins and not quiet:
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

def doCalibration(scores, labels, posclass, fileNum, showPlot, quiet):

    scores, newlabels, labels = ac.sortScoresFixLabels(scores, labels, posclass, True) # True = ascending

    maxScore  = float(max(scores))
    minScore  = float(min(scores))
    if not quiet:
        print(f'Before: min,max = {minScore},{maxScore}')
    #endif
    scores_np = (np.array(scores) - minScore) / (maxScore - minScore)
    maxScore  = float(max(scores_np))
    minScore  = float(min(scores_np))
    if not quiet:
        print(f'After: min,max = {minScore},{maxScore}')
    #endif
    numScores  = int(len(scores_np))

    #Xc_cts, Y = acKDE.getPDF(scores_np, 'epanechnikov', 'new', quiet)
    #Y1D = Y[:, 0]
    #y2 = np.interp(scores_np, Xc_cts, Y1D)
    #if showPlot:
    #   plt.plot(scores_np, y2)
    #   plt.show()
    ##endif

    for bins in [3, 6, 9]:
        if calibrationOK(numScores, bins) == 0 and (not quiet):
                print(f'Not plotted: insufficient scores ({numScores}) for {bins} bins')
        else:
            plotTitle = f'Calibration plot with {bins} bins'
            dataTitle = 'Classifier'
            if calibrationOK(numScores, bins) == 0.5 and (not quiet):
                print(f'Plotted despite insufficient scores ({numScores}) for {bins} bins')
            # endif
            params = dict(y_true=newlabels, y_prob=scores_np, normalize=False, strategy='uniform', n_bins=bins)
            if showPlot:
                fig, ax = plotCalibrationCurve(plotTitle, dataTitle, quiet, params)
            else:
                prob_predicted, prob_true = computeCalibrationCurve(plotTitle, dataTitle, quiet, params)
            #endif
            if showPlot:
                plt.show()
                fig.savefig(f'output/calib_{fileNum}-{bins}.png')
            #endif
        # endif
    # endfor
#enddef

def deepROC(testNum='1',    costs={}, sanityCheckWholeAUC=True,    showPlot=True, showData=True,
            showError=True, globalP=1, globalN=1, useCloseRangePoint=False, pAUCranges={},
            scores=[], labels=[], posclass=1, rangeAxis='FPR', quiet=False):

    doCalibration(scores, labels, posclass, testNum, showPlot, quiet)

    ep = 1*(10**-12)

    if ('rates' in costs) and costs['rates'] is True:
        skew, N, P = getSkew_withRates(labels, posclass, quiet, **costs)  # **costs includes rates
        if not quiet:
            print(f'Skew (slope) = N * N * cFPR-cTNR = {skew:0.1f}')
            print( '               -   -   ---------')
            print( '               P   P   cFNR-cTPR')
        #endif
    else:
        skew, N, P = getSkew(labels, posclass, quiet, **costs)            # **costs includes rates
        if not quiet:
            print(f'Skew (slope) = N * cFP-cTN = {skew:0.1f}')
            print( '               -   -------')
            print( '               P   cFN-cTP')
        #endif
    #endif
    if not quiet:
        print(' ')
        print(f"{'N':15s} = {N}")
        print(f"{'P':15s} = {P}")
        print(' ')
    #endif

    showThresh  = 25  # show 20 scores per plot

    # Get standard fpr, tpr, thresholds from the labels scores
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=posclass)

    # data are returned, sorted in ascending order by fpr, then tpr within fpr

    # note that this AUC may differ slightly from that obtained via full ROC
    #AUC = metrics.auc(fpr, tpr) # do not use this general auc function because it sometimes gives an error:
                                 # ValueError: x is neither increasing nor decreasing
    #AUC  = metrics.roc_auc_score(labels, scores)
    #if not quiet:
    #    print(f"{'Python AUC':15s} = {AUC:0.4f}")
    ##endif

    # Get and show the optimal ROC points (per Metz)
    opt_idx      = optimal_ROC_point_indices(fpr, tpr, skew)
    fpr_opt      = getListValuesByIndex(fpr, opt_idx)
    tpr_opt      = getListValuesByIndex(tpr, opt_idx)
    thresh_opt   = getListValuesByIndex(thresholds, opt_idx)

    if not quiet:
        for fprx, tprx, threshx in zip(fpr_opt, tpr_opt, thresh_opt):
            print(f"{'optimalROCpoint':15s} = ({fprx:0.4f},{tprx:0.4f}) at threshold: {threshx:0.4f}")
        #endfor
    #endif

    pAUCranges_copy = pAUCranges.copy()
    if sanityCheckWholeAUC == True:
        pAUCranges_copy.insert(0, [0, 1])
        indices  = np.arange(0, len(pAUCranges_copy))    # include index 0 as wholeAUC
    else:
        indices  = np.arange(1, len(pAUCranges_copy)+1)  # index 1,2,... for partial curves
    #endif
    if not quiet:
        print(f'indices: {indices}')
    #endif

    passALLeq       = True; cDelta_sum = 0; pAUC_sum  = 0; pAUCx_sum  = 0; cpAUC_sum = 0
    resultsPerGroup = []

    for pAUCrange, index in zip(pAUCranges_copy, indices):

        passEQ, cDelta, cpAUC, pAUC, pAUCx, cDeltan, cpAUCn, pAUCn, pAUCxn, extras_dict \
            = ac.do_pAUCc(mode='testing',           index=index,         pAUCrange=pAUCrange,
                          labels=labels,            scores=scores,       posclass=posclass,
                          fpr=fpr,                  tpr=tpr,             thresh=thresholds,
                          fpr_opt=fpr_opt,          tpr_opt=tpr_opt,     thresh_opt=thresh_opt,
                          numShowThresh=showThresh, testNum=testNum,     showPlot=showPlot,
                          showData=showData,        showError=showError, globalP=globalP,
                          globalN=globalN,          ep=ep,               rangeAxis=rangeAxis,
                          useCloseRangePoint=useCloseRangePoint, quiet=quiet)

        main_result_dict = dict(cDelta=cDelta,   cpAUC=cpAUC,   pAUC=pAUC,   pAUCx=pAUCx,
                                cDeltan=cDeltan, cpAUCn=cpAUCn, pAUCn=pAUCn, pAUCxn=pAUCxn,
                                passEQ=passEQ)
        main_result_dict.update(extras_dict)
        resultsPerGroup.append(main_result_dict)

        passALLeq = passALLeq and passEQ

        if index > 0:
            cDelta_sum = cDelta_sum + cDelta
            cpAUC_sum  = cpAUC_sum  + cpAUC
            pAUC_sum   = pAUC_sum   + pAUC
            pAUCx_sum  = pAUCx_sum  + pAUCx
        #endif
        if index == 0:
            AUC = extras_dict['AUC']
            c   = extras_dict['c']
        #endif
    #endfor

    # code to check for PASS here
    if sanityCheckWholeAUC:
        pass1     = ac.areEpsilonEqual(cDelta_sum, c,   'cDelta_sum', 'c',   ep, quiet)
    #endif
    pass2     = ac.areEpsilonEqual(cpAUC_sum,  AUC, 'cpAUC_sum',  'AUC', ep, quiet)
    pass3     = ac.areEpsilonEqual(pAUC_sum ,  AUC, 'pAUC_sum',   'AUC', ep, quiet)
    pass4     = ac.areEpsilonEqual(pAUCx_sum,  AUC, 'pAUCx_sum',  'AUC', ep, quiet)
    if sanityCheckWholeAUC:
        passALLeq = passALLeq and pass1 and pass2 and pass3 and pass4
    else:
        passALLeq = passALLeq and pass2 and pass3 and pass4
    #endif

    return resultsPerGroup, passALLeq
#enddef
