# do_pAUCc.py
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
#   Modifications by Andre Carrington, 2020
#       indexing and specifying partial curves is clearer and simpler
#       added concordance matrix plot
#       added calculation of standardized partial area and partial area index
#       better annotation at the (0,0) and (1,1) points
#       shows skew used for optimal ROC point and shows multiple optimal points
#
# Functions:
#    c_statistic
#    average_measures_discrete
#    concordant_partial_AUC
#    partial_c_statistic
#    partial_area_index_proxy
#    standardized_partial_area_proxy
#    interpolateROC
#    get_Match_or_Interpolation_Points
#    getRange_pAUCc
#    getRange_cDelta
#        interpolateAxis (internal function)
#    getFullROC
#        addStandardROCpoint (internal function)
#        addROCtie (internal function)
#    get_cMatrix_Label_Size_Fontsize
#    get_ROC_Curve_Label_Offset_Fontsize
#    plotConcordanceMatrix
#    plotROC
#    plotSimpleROC
#    plotOpt
#    plotPartialArea
#    addPoints
#    makeLabels01
#    rocErrorCheck
#    cSanityCheck
#    cErrorCheck
#    areEpsilonEqual
#    Heaviside
#    get_plabel
#    do_pAUCc
#
# Compute and show whole and partial AUC measures as discussed in Carrington et al [1].
#    Whole measures:     AUC,        c
#    Partial measures:   pAUCc,      cdelta,      pAUC (pAUCy), pAUCx,      lc
#    Normalized partial: pAUCc_norm, cdelta_norm, pAUC_norm,    pAUCx_norm, lc_norm
#
# References:
# 1. Carrington AM, Fieguth P, Qazi H, Holzinger A, Chen H, Mayr F
#    Manuel D. A new concordant partial AUC and partial c statistic
#    for imbalanced data in the evaluation of machine learning
#    algorithms. BMC Medical Informatics and Decision Making, 2020.
#
import numpy                    as     np
import matplotlib.pyplot        as     plt
import matplotlib.ticker        as     ticker
from   sklearn                  import metrics
#from   c_statistic        import concordanceForAUC
#from   partialConcordanceForAUC import partialConcordanceForAUC
#import time
import math

def c_statistic(posScores, negScores):
    ''' c_statistic computes the cStatistic given a vector of scores for actual positives
        and a vector of scores for actual negatives in the ROC data '''
    P = len(posScores)
    N = len(negScores)
    c = 0
    for j in range(0, P):
        for k in range(0, N):
            c = c + Heaviside(posScores[j]-negScores[k])
        #endfor
    #endfor
    c = c / (P * N)
    return c
#enddef

def average_measures_discrete(pfpr, ptpr, plabel):
    # This function requires points generated from the getFullROC function
    sumSensFP = 0
    sumSpecTP = 0
    delx      = 0  # the weight or area in this case is a discrete count
    dely      = 0  # the weight or area in this case is a discrete count

    #for fpr, tpr, label in zip(afpr, atpr, alabel):
    for fpr, tpr, label in zip(pfpr, ptpr, plabel):
        if label == 0 and tpr > 0:        # neg = FP. for sens: the first point has no area/weight
            sumSensFP = sumSensFP + tpr
            delx      = delx + 1
        elif label == 1 and (1-fpr) > 0:  # pos = TP. for spec: the last  point has no area/weight
            sumSpecTP = sumSpecTP + (1-fpr)
            dely      = dely + 1
        #endif
    #endfor
    if delx > 0:
        avgSensFP = (1/delx) * sumSensFP
    else:
        avgSensFP = 0
    #endif
    if dely > 0:
        avgSpecTP = (1/dely) * sumSpecTP
    else:
        avgSpecTP = 0
    #endif
    return avgSensFP, avgSpecTP
#enddef

def concordant_partial_AUC(pfpr, ptpr):
    ''' Computes the concordant partial area under the curve and alternatives, given arrays of \n
    "partial fpr" and "partial tpr" values.  These arrays only contain points on the partial curve \n
    and the trapezoidal rule is used to compute areas with these points. \n
    \n
    pAUCc:      the concordant partial area under the curve \n
    pAUC:       the (vertical) partial area under the curve \n
    pAUCx:      the horizontal partial area under the curve \n

    pAUCc_norm: the concordant partial area under the curve normalized \n
    pAUC_norm:  the (vertical) partial area under the curve normalized \n
    pAUCx_norm: the horizontal partial area under the curve normalized \n
    '''

    # xrange [a,b]
    a    = float(pfpr[0])
    b    = float(pfpr[-1])
    delx = b - a
    vertical_stripe_area = (1 * delx)

    # yrange [f,g]
    f    = float(ptpr[0])
    g    = float(ptpr[-1])
    dely = g - f
    horizontal_stripe_area = (dely * 1)

    if delx == 0:
        print("Warning: For pAUC and pAUCc the width (delx) of the vertical column is zero.")
        pAUC  = 0
        pAUCn = 0
        sPA   = 0
    else:
        # Compute the partial AUC mathematically defined in (Dodd and Pepe, 2003) and conceptually defined in
        #   (McClish, 1989; Thompson and Zucchini, 1989). Use the trapezoidal rule to compute the integral.
        pAUC  = np.trapz(ptpr, pfpr)  # trapz is y,x
        pAUCn = pAUC / vertical_stripe_area
    #endif

    if dely == 0:
        print("Warning: For pAUCx and pAUCc the height (dely) of the horizontal stripe is zero.")
        pAUCx  = 0
        pAUCxn = 0
    else:
        # Compute the horizontal partial AUC (pAUCx) defined in (Carrington et al, 2020) and as
        # suggested by (Walter, 2005) and similar to the partial area index (PAI)
        # (Nishikawa et al, ?) although PAI has a fixed right boundary instead and a slightly
        # different mathematical definition.
        #
        # Normally we would compute the area to the right of the curve, the horizontal integral,
        # as follows:
        #   1. swap the axes
        #   2. flip the direction of the new vertical
        #   3. compute the (vertical) integral
        #
        tempX  = ptpr                      # swap the axes
        tempY  = list(1 - np.array(pfpr))  # flip the direction
        pAUCx  = np.trapz(tempY, tempX)    # trapz is y,x
        pAUCxn = pAUCx / horizontal_stripe_area
    #endif

    total_for_norm = vertical_stripe_area + horizontal_stripe_area
    if total_for_norm == 0:
        pAUCc  = 0
        pAUCcn = 0
        print('Warning: Zero length partial curve specified.')
    else:
        pAUCc  = (1/2)*pAUCx + (1/2)*pAUC  # the complexity is in the derivation, meaning/generalization
        #   and relation to the c and partial c statistic, not the
        #   formula which looks like a simple average
        #pAUCcn= (pAUCx + pAUC) / total_for_norm
        if vertical_stripe_area > 0 and horizontal_stripe_area > 0: # NEW part-wise normalization
            pAUCcn = (1 / 2) * (pAUC  / vertical_stripe_area) + (1 / 2) * (pAUCx / horizontal_stripe_area)
        elif vertical_stripe_area   == 0:
            pAUCcn = (1 / 2) * (pAUCx / horizontal_stripe_area)
        elif horizontal_stripe_area == 0:
            pAUCcn = (1 / 2) * (pAUC  / vertical_stripe_area)
        # endif
    #endif

    return pAUCc, pAUC, pAUCx, pAUCcn, pAUCn, pAUCxn
#enddef

def partial_c_statistic(posScores, negScores, posWeights, negWeights):
    ''' partial_c_statistic computes the cStatistic given a vector of scores for actual
        positives and a vector of scores for actual negatives in the ROC data and
        corresponding weight vectors with elements in [0,1] that indicate which instances
        are in the partial range and how much weight for each instance.  That is,
        boundary instances may have partial weight, interior instances will have weight 1
        and exterior instances will have weight 0'''
    P      = len(posScores)
    N      = len(negScores)
    Pw     = float(sum(posWeights))
    Nw     = float(sum(negWeights))
    cp     = 0
    cn     = 0
    cLocal = 0
    for j in range(0, P):
        for k in range(0, N):
            h      = Heaviside(float(posScores[j]) - float(negScores[k]))
            cp     = cp     + float(posWeights[j]) * h
            cn     = cn     + float(negWeights[k]) * h
            cLocal = cLocal + float(posWeights[j]) * float(negWeights[k]) * h
        #endfor
    #endfor
    cDelta  = cp/(2*P*N)  + cn/(2*P*N)
    #cDeltan= (cDelta * (2*P*N)) / ((N*Pw) + (P*Nw))  # OLD overall normalization
    if   Pw > 0 and Nw > 0:
        cDeltan = cp/(2*N*Pw) + cn/(2*P*Nw)           # NEW part-wise normalization
    elif Pw == 0:
        cDeltan = cn/(2*P*Nw)
    elif Nw == 0:
        cDeltan = cp/(2*N*Pw)
    #endif

    if Pw*Nw == 0:
        cLocal = 0
    else:
        cLocal = cLocal / (Pw*Nw)
    #endif

    return cDelta, cDeltan, cLocal
#enddef

def partial_area_index_proxy(pfpr, ptpr):
    # this is not the full binormal curve fit version of Jiang et al.'s partial area index
    # it is instead a proxy for that using the simpler definition of the area to the right of a curve

    # xrange [a,b] should be [a,1]
    b = float(pfpr[-1])

    # yrange [f,g]
    f        = float(ptpr[0])
    g        = float(ptpr[-1])
    dely_PAI = 1 - f  # in PAI the right boundary g is fixed at 1
    PAI_stripe_area = (dely_PAI * 1)

    if b != 1 or g != 1:
        raise ValueError('PAI requires data up to and including ROC point (1,1)')
    #endif

    if dely_PAI == 0:
        print(" Warning: For PAI the height (dely) of the horizontal stripe is zero.")
        PAI = 0
    else:
        # Compute the partial area index (PAI) (Nishikawa et al, ?) similar to pAUCx, but with a
        # fixed right boundary. Compute the area to the right of the curve, the horizontal integral,
        # using the trapezoid rule, as follows:
        #   1. swap the axes
        #   2. flip the direction of what originally was the FPR/x-axis
        #      because for area to the right of the curve
        #      up (or the top of the curve) is to the left (toward 0)
        #   3. compute the (vertical) integral with the trapezoid rule
        tempX = ptpr
        tempY = list(1 - np.array(pfpr))
        PAI   = np.trapz(tempY, tempX)    # trapz is y,x
        PAI   = PAI / PAI_stripe_area
    # endif
    return PAI
#enddef

def standardized_partial_area_proxy(pfpr, ptpr):
    # this is not the full binormal curve fit version of McClish's sPA
    # it is instead a proxy for that using a simpler definition
    '''
    sPA:        the (vertical) partial area standardized, subtracts trapezoid below major diagonal
    '''
    # xrange [a,b]
    a    = float(pfpr[0])
    b    = float(pfpr[-1])
    delx = b - a
    vertical_stripe_area = (1 * delx)

    # yrange [f,g]
    f    = float(ptpr[0])
    g    = float(ptpr[-1])
    dely = g - f
    horizontal_stripe_area = (dely * 1)

    if delx == 0:
        print("Warning: For sPA the width of the region is zero.")
        sPA   = 0
    else:
        # Compute the standardized partial area (sPA) defined in (McClish, 2006), with a correction.
        # McClish calls Atrapezoid: Amin, but McClish erroneously swapped the Amin & Amax formulas
        # McClish calls Acolumn:    Amax

        sPA_rescale = lambda x: 0.5 * (1 + x)  # sPA rescales [0,1] to [0.5,1]
        pAUC        = np.trapz(ptpr, pfpr)  # trapz is y,x
        Atrapezoid  = (0.5 * (a + b)) * (b - a)  # average height of trap  * width
        Acolumn     =               1 * (b - a)  #                       1 * width
        sPA         = sPA_rescale( (pAUC-Atrapezoid)/(Acolumn-Atrapezoid) )
    #endif
    return sPA
#enddef

def interpolateROC(rstat, ostat, thresh, ixL, ixR, interpValue):
    ''' returns new_ostat_value, new_thresh_value '''
    # STATUS: COMPLETE
    #
    # given points to the left and right at indices ixL and ixR
    # interpolate the data (statistics)...
    # rstat = range statistics (e.g., FPR; or TPR)
    # ostat = other statistics (e.g., TPR, if rstat is FPR; or FPR)

    # get widths for interpolation
    width_rstat_RL   = rstat[ixR]   - rstat[ixL]
    width_ostat_RL   = ostat[ixR]   - ostat[ixL]
    width_thresh_RL  = thresh[ixR]  - thresh[ixL]
    perCentFromL     = (interpValue - rstat[ixL])  / width_rstat_RL

    # interpolate
    # new_rstat_value  = interpValue
    new_ostat_value  = ostat[ixL]   + perCentFromL * width_ostat_RL
    new_thresh_value = thresh[ixL]  + perCentFromL * width_thresh_RL

    return new_ostat_value, new_thresh_value, perCentFromL
#enddef

def get_Match_or_Interpolation_Points(rstat, endpoint):
    ''' returns ix, ixL, ixR '''

    ix = np.argwhere(rstat == endpoint).flatten()  # find matching point, as a flat list
    if len(ix) == 0: # if no matching indices in ix, then interpolate
        # to interpolate find points to the right and left
        ixR = np.argwhere(rstat > endpoint).flatten()  # find points to the right, as a flat list
        if len(ixR) == 0:   # if there are no points to the right, then error
            raise ValueError('Cannot find an ROC point right of value to interpolate')
        else:
            ixR = ixR[0]        # take the closest/first point to the right
        #endif

        ixL = np.argwhere(rstat < endpoint).flatten()  # find points to the left, as a flat list
        if len(ixL) == 0:   # if there are no points to the left, then error
            raise ValueError('Cannot find an ROC point left of value to interpolate')
        else:
            ixL = ixL[-1]       # take the closest/last point to the left
        #endif
        # ix is empty (no exact match)
        # ixL and ixR are not empty

    else:  # exact match found
        # ix is not empty
        ixR = []
        ixL = []
    #endif
    return ix, ixL, ixR
#enddef

def getRange_pAUCc(ffpr, ftpr, fthresh, rangeAxis1, rangeEndpoints1, rocRuleLeft, rocRuleRight):
    #                                   FPR or TPR; in FPR or TPR;
    ''' returns pfpr, ptpr, rangeEndpoints2, rangeEndpoints0, rangeIndices0 \n
    # \n
    # This function takes as input: \n
    #    1. Full ROC data (not just standard ROC data) from a prior call to                 \n
    #       getFullROC: ffpr, ftpr, fthresh. One element per instance in data               \n
    # \n
    #    2. a range (rangeEndpoints1) in FPR or TPR (rangeAxis1)             (0.1,0.8), FPR \n
    # \n
    #    3. and rules (rocRuleLeft, rocRuleRight) that resolve ambiguous                    \n
    #       range endpoints that match multiple vertical or horizontal ROC                  \n
    #       points \n
    # \n
    # This function outputs: \n
    #    1. The partial curve in pfpr and ptpr. \n
    # \n
    #    2. The range of values in the other axis (rangeEndpoints2)          (0.2, 0.7), TPR \n
    # \n
    #    3. The range of threshold values (rangeEndpoints0)                  (0.95, 0.1)     \n
    #       note: thresholds can be [-inf,inf] not just [0,1]                or (1.7, -0.6)  \n
    #       and the range of indices (rangeIndices0) for ffpr, fptr, fthresh [2, 6]          \n
    #       and approx indices when interpolated instead of match (approxIndices0) [2, 6]    \n
    '''
    #
    FPR, TPR, NE, SW = ('FPR', 'TPR', 'NE', 'SW')
    # error checks
    rocErrorCheck(ffpr, ftpr, fthresh, rangeEndpoints1, rangeAxis1, rocRuleLeft, rocRuleRight)

    pfpr_np = np.array(ffpr)
    ptpr_np = np.array(ftpr)
    n       = len(ffpr)

    if   rangeAxis1 == FPR:
        rstat = ffpr.copy()  # range statistic
        ostat = ftpr.copy()  # other statistic
    else:  # rangeAxis1 == TPR:
        rstat = ftpr.copy()  # range statistic
        ostat = ffpr.copy()  # other statistic
    #endif

    rangeIndices0   = ['NA' , 'NA' ]  # initialize to detectable nonsense value
    approxIndices0  = ['NA' , 'NA' ]  # initialize to detectable nonsense value
    rangeEndpoints0 = ['NA' , 'NA' ]  # initialize to detectable nonsense value
    rangeEndpoints2 = ['NA' , 'NA' ]  # initialize to detectable nonsense value
    # we need to process the endpoints in right to left order [1, 0] so that
    # when we delete points from pfpr and ptpr, the indices on the left still
    # make sense even if we have changed the indices on the right.  In the
    # other direction both indices are affected.
    indices_reversed         = [1, 0]
    rangeEndpoints1_reversed = [rangeEndpoints1[1], rangeEndpoints1[0]]
    rocRules_reversed        = [rocRuleRight, rocRuleLeft]
    for endpoint, i, rocRule in zip(rangeEndpoints1_reversed, indices_reversed, rocRules_reversed):

        # find indices ix which match rangeroc[0]
        ix, ixL, ixR = get_Match_or_Interpolation_Points(rstat, endpoint)

        if len(ix) == 0:  # if no exact match, then interpolate
            rangeEndpoints2[i], rangeEndpoints0[i], perCentFromLeft = \
                interpolateROC(rstat, ostat, fthresh, ixL, ixR, endpoint)
            if perCentFromLeft > 0.5:
                approxIndices0[i] = ixR
            else:
                approxIndices0[i] = ixL
            #endif
            print(f'Interpolating {rangeAxis1}[{i}] between {rstat[ixL]:0.3f} and {rstat[ixR]:0.3f}')
            #   with a newly interpolated point at rangeEndpoints0
            if rangeAxis1 == FPR:
                if i == 1:    # right/top
                    pfpr_np = np.delete(pfpr_np, np.arange(ixR, n))
                    ptpr_np = np.delete(ptpr_np, np.arange(ixR, n))
                    pfpr_np = np.append(pfpr_np, endpoint)
                    ptpr_np = np.append(ptpr_np, rangeEndpoints2[i])
                elif i == 0:  # left/bottom
                    pfpr_np = np.delete(pfpr_np, np.arange(0, ixL))
                    ptpr_np = np.delete(ptpr_np, np.arange(0, ixL))
                    pfpr_np = np.insert(pfpr_np, 0, endpoint)
                    ptpr_np = np.insert(ptpr_np, 0, rangeEndpoints2[i])
                #endif
            else:  # rangeAxis1 == TPR:
                if i == 1:    # right/top
                    pfpr_np = np.delete(pfpr_np, np.arange(ixR, n))
                    ptpr_np = np.delete(ptpr_np, np.arange(ixR, n))
                    pfpr_np = np.append(pfpr_np, rangeEndpoints2[i])
                    ptpr_np = np.append(ptpr_np, endpoint)
                elif i == 0:  # left/bottom
                    pfpr_np = np.delete(pfpr_np, np.arange(0, ixL))
                    ptpr_np = np.delete(ptpr_np, np.arange(0, ixL))
                    pfpr_np = np.insert(pfpr_np, 0, rangeEndpoints2[i])
                    ptpr_np = np.insert(ptpr_np, 0, endpoint)
                #endif
            # endif
        else:  # found one or more indices in ix that match endpoint
               # use rules to choose which of multiple matching points to use
               # (this logic also works for a single matching point)
            if rocRule == SW:       # take earliest point
                ix_to_use = ix[0]
            else:  # rocRule == NE: # take last point
                ix_to_use = ix[-1]
            #endif
            rangeIndices0[i]    =         ix_to_use
            rangeEndpoints0[i]  = fthresh[ix_to_use]
            rangeEndpoints2[i]  =   ostat[ix_to_use]
            if i == 1:    # right/top
                if ix_to_use < n - 1:  # if not last  instance then truncate right part
                    pfpr_np = np.delete(pfpr_np, np.arange(ix_to_use + 1, n))
                    ptpr_np = np.delete(ptpr_np, np.arange(ix_to_use + 1, n))
                #endif
            elif i == 0:  # left/bottom
                if ix_to_use > 0:      # if not first instance then truncate left part
                    pfpr_np = np.delete(pfpr_np, np.arange(0, ix_to_use))
                    ptpr_np = np.delete(ptpr_np, np.arange(0, ix_to_use))
                # endif
            #endif
        #endif
    #endfor
    pfpr = pfpr_np.tolist()
    ptpr = ptpr_np.tolist()
    return pfpr, ptpr, rangeEndpoints2, rangeEndpoints0, rangeIndices0, approxIndices0
# enddef

def getRange_cDelta(ffpr, ftpr, fthresh, scores, labels, posclass, xrange, yrange):
    '''
    # returns: negIndexC, posIndexC, negScores, posScores, negWeights, posWeights \n
    # \n
    # This function requires understanding of the concordance matrix (Carrington et \n
    # al, 2020; Hilden, 1991) \n
    # \n
    # This function takes as input: \n
    #    1. Full ROC data (not just standard ROC data) from a prior call to                 \n
    #       getFullROC: ffpr, ftpr, fthresh. One element per instance in data               \n
    # \n
    #    2. The scores and labels (scores, labels) that create ROC data, and                \n
    #       identification of which label is positive (posclass)                            \n
    # \n
    #    3. The range in FPR and TPR (xrange, yrange) that uniquely define the              \n
    #       partial curves endpoints.                                                       \n
    # \n
    # This function outputs: \n
    #    1. indices (negIndexC, posIndexC) to the instances   [0, 2], [1, 4]                        \n
    #       (negScores, posScores) along each axis.           FPR: [0.8, 0.55, 0.4, 0.3]            \n
    #                                                         TPR: [1.1, 0.95, 0.7, 0.6, 0.5, 0.48] \n
    #       and their weights (negWeights, posWeights)        [0.7, 1, 1, 0], [0, 1, 1, 1, 0.2, 0]  \n
    '''
    # check inputs for errors
    n     = len(fthresh)
    if n != len(ffpr) and n != len(ftpr):
        return "Error, fpr and tpr and thresh must have the same length"
    #cErrorCheck(cRuleLeft, cRuleRight)
    cSanityCheck(xrange)
    cSanityCheck(yrange)

    # we will obtain the positive and negative instances separately in sorted
    # order from highest value near ROC (0,0) to lowest value at the furthest
    # value therefrom.  See the concordance matrix in Carrington et al, 2020.

    # use newlabels to make sure the positive class has a higher label value
    newlabels  = makeLabels01(labels, posclass)

    # sort in descending order by scores, then by newlabels (within tied scores)
    # (we must do this ascending, and then reverse it)
    dtype2     = [('scores', float), ('newlabels', int), ('labels', int)]  # assumes labels are int
    rocTuples2 = list(zip(scores, newlabels, labels))
    rocArray2  = np.array(rocTuples2, dtype=dtype2)
    temp2      = np.sort(rocArray2,   order=['scores', 'newlabels'])
    final2     = temp2[::-1]        # reverse it
    # put the sorted data back into the original list variables
    scores, newlabels, labels = zip(*final2)  # zip returns (immutable) tuples
    # we could convert these to lists, but we don't need to: scores = list(scores)

    # get the positive instances (their scores) along the TPR axis
    posIdx     = np.argwhere(np.array(newlabels) == 1).flatten()
    posScores  = []
    for ix in posIdx:
        posScores = posScores + [scores[ix]]
    #endfor

    # get the negative instances (their scores) along the FPR axis
    negIdx     = np.argwhere(np.array(newlabels) == 0).flatten()
    negScores  = []
    for ix in negIdx:
        negScores = negScores + [scores[ix]]
    #endfor

    # initialize pos and neg indices and weights
    ptotal     = len(posScores)
    ntotal     = len(negScores)
    posWeights = np.ones((ptotal, 1), float)
    negWeights = np.ones((ntotal, 1), float)
    posWeight  = np.zeros((2, 1), float)
    negWeight  = np.zeros((2, 1), float)
    negIndexC  = np.zeros((2, 1), int)    # or np.int, it doesn't matter which
    posIndexC  = np.zeros((2, 1), int)

    # get index of the left boundary instance and the updated weights that
    # interpolate that instance (where weight is left-to-right thinking)
    def interpolateAxis(axisValue,numAxisInstances):
        rawIndex    = axisValue * numAxisInstances  # rawIndex has a decimal value, e.g. 1.2
        indexBefore = math.floor(rawIndex)          # 0-based index
        if indexBefore == rawIndex and indexBefore > 0:
            indexBefore = indexBefore - 1
        weight      = rawIndex - indexBefore
        return indexBefore, weight
    #enddef

    negIndexC[0], negWeight[0] = interpolateAxis(xrange[0], ntotal)
    negIndexC[1], negWeight[1] = interpolateAxis(xrange[1], ntotal)
    if negIndexC[0] == negIndexC[1]:
        # this case is complicated: we are interpolating twice, both the
        # top and bottom boundaries in the same instance
        negWeights[negIndexC[0]] = negWeight[1] - negWeight[0] # set weight at boundary
    else:
        # interpolated weight is based on left to right thinking (0 at left, 1 at right)
        # but the weight we need for a left boundary is opposite, so
        negWeight0_reversed      = 1 - negWeight[0]
        negWeights[negIndexC[0]] = negWeight0_reversed  # set weight at boundary
        negWeights[negIndexC[1]] = negWeight[1]         # set weight at boundary
    #endif
    # cast single element numpy array to int, for range/slice to work
    negWeights[0: int(negIndexC[0])]        = 0  # zeroize left of left boundary
    negWeights[int(negIndexC[1])+1: ntotal] = 0  # zeroize right of right boundary

    posIndexC[0], posWeight[0] = interpolateAxis(yrange[0], ptotal)
    posIndexC[1], posWeight[1] = interpolateAxis(yrange[1], ptotal)
    if posIndexC[0] == posIndexC[1]:
        # this case is complicated: we are interpolating twice, both the
        # top and bottom boundaries in the same instance
        posWeights[posIndexC[0]] = posWeight[1] - posWeight[0]  # set weight at boundary
    else:
        # interpolated weight is based on bottom to top thinking (0 at bottom, 1 at top)
        # but the weight we need for a bottom boundary is opposite, so
        posWeight0_reversed      = 1 - posWeight[0]
        posWeights[posIndexC[0]] = posWeight0_reversed  # set weight at boundary
        posWeights[posIndexC[1]] = posWeight[1]         # set weight at boundary
    #endif
    # cast single element numpy array to int, for range/slice to work
    posWeights[0: int(posIndexC[0])]        = 0  # zeroize below bottom boundary
    posWeights[int(posIndexC[1])+1: ptotal] = 0  # zeroize above top boundary

    return negIndexC, posIndexC, negScores, posScores, negWeights, posWeights
# enddef

def getFullROC(labels, scores, posclass):
    # Get the "full" set of points in an ROC curve that correspond to each
    # instance in the data.  This differs from the "standard" ROC points and
    # procedure as outlined in Fawcett, because that procedure skips points
    # which are redundant for a "standard" ROC plot.
    #
    # What are skipped points? In the standard empirical ROC procedure, which
    # creates a staircase plot, if an ROC curve "stair" ascends 3 times,
    # or moves horizontally 4 times, or moves diagonally (tied scores) 3 times,
    # then these intermediary points are not included in the "standard" ROC
    # procedure, with one small exception: points adjacent to (0,0) and (1,1).
    #
    # The "full" ROC procedure is necessary in 2 scenarios:
    #   1) to compute the partial c statistic, and/or
    #   2) to show the full set of actual threshold values along the ROC
    #      curve (in a plot)--which is informative for decision-making.
    #
    # It is necessary because the thresholds at intermediary points do not
    # in general, change in a linear manner, as liner interpolation would infer
    # in the first case for partial c statistics, or as a user may infer in
    # the second case.
    #
    # Notably, the concordant partial AUC (or other partial area measures)
    # work the same with either "full" or "standard" ROC procedures.
    #
    # In variable names, the letter "f" represents "full": e.g., ffpr, ftpr.

    # use newlabels to make sure the positive class has a higher label value
    newlabels = makeLabels01(labels, posclass)

    # sort in descending order by scores, then by newlabels (within tied scores)
    # (we must do this ascending, and then reverse it)
    dtype     = [('scores', float), ('newlabels', int), ('labels', int)]  # assumes labels are int
    rocTuples = list(zip(scores, newlabels, labels))
    rocArray  = np.array(rocTuples, dtype=dtype)
    temp      = np.sort(rocArray,   order=['scores', 'newlabels'])
    final     = temp[::-1]        # reverse it
    # put the sorted data back into the original list variables
    scores, newlabels, labels = zip(*final)

    n          = len(labels)+1  # +1 for the (0,0) point
    finalIndex = n - 1
    blank      = np.zeros(n)
    ffpr, ftpr, fthresh = (blank.copy(), blank.copy(), blank.copy())
    fnewlabel  = np.array(newlabels)

    # for score, newlabel, label in zip(scores, newlabels, labels)
    thisFP  = 0
    thisTP  = 0
    numTP   = len(np.where(np.array(newlabels) == 1)[0])
    numFP   = len(np.where(np.array(newlabels) == 0)[0])
    tPrev   = math.inf   # previous threshold
    numTies       = 0
    firstTieIndex = 0

    def addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, tPrev):
        ftpr[index]    = thisTP / numTP  # first time here is (0, 0) with thresh Inf
        ffpr[index]    = thisFP / numFP
        fthresh[index] = tPrev
        return ffpr, ftpr, fthresh
    #enddef

    def addROCtie(ffpr, ftpr, fthresh, index, tPrev, numTies, firstTieIndex):
        rise = ftpr[index] - ftpr[firstTieIndex-1]  # incl. point before
        run  = ffpr[index] - ffpr[firstTieIndex-1]  # incl. point before
        for i in np.arange(0, numTies):
            ftpr[firstTieIndex + i]    = ftpr[firstTieIndex-1] + rise * ((i+1) / numTies)
            ffpr[firstTieIndex + i]    = ffpr[firstTieIndex-1] + run  * ((i+1) / numTies)
            fthresh[firstTieIndex + i] = tPrev
        # endfor
        return ffpr, ftpr, fthresh
    #enddef

    index = 0  # index of all/full ROC points including hidden ties
    # add (0,0) point
    ffpr, ftpr, fthresh = addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, tPrev)
    np.insert(fnewlabel, 0, -1)
    index = 1

    for score, newlabel, label in final:
        # we move an imaginary threshold (thresh) from the highest to lowest score
        # which is left to right in ROC, from the all negative threshold at (0,0)
        # to the all positive threshold at (1,1)
        #
        # for any given point after (0,0), it is either:
        #   (a) the last roc point, not tied w next, follow remaining logic below
        #   (b) a score not tied w previous or next
        #   (c) a new tied score (not tied w previous, but tied w next)
        #   (d) a middle tied score (tied w previous, and next)
        #   (e) a last tied score (tied w previous, but not next)
        #
        if newlabel == 1:
            thisTP  = thisTP + 1
        else:
            thisFP  = thisFP + 1
        #endif
        #
        if index == finalIndex:                     # (a) last roc point
            tied_w_next = False
        else:
            tied_w_next = (score == scores[index])  # not "index + 1" because scores has no (0,0) point
        #endif
        tied_w_prev = (score == tPrev)
        if   not tied_w_prev and not tied_w_next:   # (b) not a tie
            ffpr, ftpr, fthresh = addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, score)
            tPrev   = score       # set previous threshold for next iteration to current score
            numTies = 0
        elif not tied_w_prev and     tied_w_next:   # (c) new tie
            ffpr, ftpr, fthresh = addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, score)
            firstTieIndex = index
            numTies = 1
            tPrev   = score       # set previous threshold for next iteration to current score
        elif     tied_w_prev and     tied_w_next:   # (d) middle tie
            numTies = numTies + 1
        elif     tied_w_prev and not tied_w_next:   # (e) last tie (in current series)
            ffpr, ftpr, fthresh = addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, score)
            numTies = numTies + 1
            ffpr, ftpr, fthresh = addROCtie(ffpr, ftpr, fthresh, index, score, numTies, firstTieIndex)
        #endif
        index = index + 1  # increment index for next iteration
    #endfor
    ffpr    = ffpr[:index+1].copy()
    ftpr    = ftpr[:index+1].copy()
    fthresh = fthresh[:index+1].copy()
    return ffpr, ftpr, fthresh, fnewlabel
#enddef

def get_cMatrix_Label_Size_Fontsize(val):
    ''' returns label, sizexy, fontsize '''

    # label offsets are dependent on the fontsize
    fontsize      = 'x-small'
    # fontsize: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    # number formatting
    if val < 10:
        # label   = "{:.2f}".format(val)
        label     = f'{val:.2g}'
    elif np.isinf(val):
        if val > 0:
            label = 'inf'
        else:
            label = '-inf'
        #endif
    else:
        val       = int(round(val))
        label     = f'{val:.2d}'
    #endif
    if val < 1 and val > 0:   # if decimal included
        numberWidth  = 5 * (len(label)-1)
        decimalWidth = 1
        width        = numberWidth + decimalWidth
    else:
        width  = 5 * len(label)
    #endif
    sizexy = (width, 5)
    return label, sizexy, fontsize
#enddef

def get_ROC_Curve_Label_Offset_Fontsize(x, y, t, maxThresh, fancyLabel):

    if 'fancyLabel' not in locals():
        fancyLabel = False

    inf_symbol = '\u221e'
    fontsize   = 'x-small'
    # fontsize: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    # label offsets are dependent on the fontsize

    # setup prefix and suffix for (0,0)
    if math.isinf(t) and t > 0:  # for positive infinity threshold
        t      = maxThresh       # replace it with greater than maxThresh
        if fancyLabel:
            prefix = '('; suffix = f',{inf_symbol}]'
        else:
            prefix = '>'; suffix = ''
    else:
        prefix = ''; suffix = ''
    #endif

    # setup prefix and suffix for (1,1)
    if x == 1 and y == 1:
        if fancyLabel:
            prefix = f'[-{inf_symbol},'; suffix = ']'
        else:
            prefix = ''; suffix = ''
    #endif

    # number formatting
    if t < 10:
        # label    = "{:.2f}".format(t)
        label = f'{prefix}{t:.2g}{suffix}'
    else:
        t          = int(round(t))
        label = f'{prefix}{t:.2d}{suffix}'
    #endif

    # setup width and offset
    if x == 1 and y == 1:
        if t < 1 and t > 0:   # if decimal included
            numberWidth  = 5 * (len(label)-1)
            decimalWidth = 1
            if fancyLabel:
                width    = numberWidth + decimalWidth -10
            else:
                width    = numberWidth + decimalWidth
        else:
            if fancyLabel:
                width    = 5 * len(label) - 10
            else:
                width    = 5 * len(label)
        #endif
        if fancyLabel:
            offset = (-width, 2)
        else:
            offset = (-width, -7)
    elif x == 0 and y == 0:
        offset = (1, 2)
    else:
        offset = (1, -7)
    # endif
    return label, offset, fontsize
#enddef

def plotConcordanceMatrix(fpr, tpr, negScores, posScores, plotTitle, maxInstancesPerAxis):
    ''' returns fig, ax '''

    # ordinal function from:
    # https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement  which adapts Gareth's answer from:
    # https://codegolf.stackexchange.com/questions/4707/outputting-ordinal-numbers-1st-2nd-3rd#answer-4712
    ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

    negScores = list(negScores)
    posScores = list(posScores)
    ntotal    = len(negScores)
    ptotal    = len(posScores)
    nfactor   = 1
    pfactor   = 1
    nsuffix   = ''
    psuffix   = ''

    if ntotal > maxInstancesPerAxis:
        nfactor    = ntotal/maxInstancesPerAxis
        nfactortxt = int(round(nfactor))
        nsuffix = f' (approx. every {ordinal(nfactortxt)})'
    #endif
    if ptotal > maxInstancesPerAxis:
        pfactor    = ptotal/maxInstancesPerAxis
        pfactortxt = int(round(pfactor))
        psuffix = f' (approx. every {ordinal(pfactortxt)})'
    #endif
    if nfactor > 1 and pfactor > 1:
        maxfactor    = max(nfactor, pfactor)
        maxfactortxt = int(round(maxfactor))
        nfactor      = maxfactor
        pfactor      = maxfactor
        nsuffix = f' (approx. every {ordinal(maxfactortxt)})'
        psuffix = f' (approx. every {ordinal(maxfactortxt)})'
    #endif

    show_ntotal = int(math.floor(ntotal/nfactor)) # in the plot shown we round-off (ignore) a bit at the end
    show_ptotal = int(math.floor(ptotal/pfactor)) # in the plot shown we round-off (ignore) a bit at the end

    # create plot with ROC curve
    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1, xticklabels=[], yticklabels=[])
    plt.plot(fpr, tpr, color='blue', lw=2)

    # label negative instances
    for i in range(0, show_ntotal):
        idx   = int(round(i*nfactor))
        x     = (i+0.5)/show_ntotal      # half indicates the center of the column in the concordance matrix
        score = float(negScores[idx])
        label, sizexy, fontsize = get_cMatrix_Label_Size_Fontsize(score)
        offset    = (x-(0.5*sizexy[0]), -2-sizexy[1])
        plt.annotate(label, (x, 0), textcoords="offset points",
                     xytext=offset, ha='left', fontsize=fontsize)
    #endfor

    # label positive instances
    for i in range(0, show_ptotal):
        idx   = int(round(i*pfactor))
        y     = (i+0.5)/show_ptotal      # half indicates the center of the column in the concordance matrix
        score = float(posScores[idx])
        label, sizexy, fontsize = get_cMatrix_Label_Size_Fontsize(score)
        offset    = (-2-sizexy[0], y-(0.5*sizexy[1]))
        plt.annotate(label, (0, y), textcoords="offset points",
                     xytext=offset, ha='left', fontsize=fontsize)
    #endfor

    plt.xlabel(f' \nNegative Instances{nsuffix}')
    plt.ylabel(f'Positive Instances{psuffix}\n \n ')
    plt.title(plotTitle)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    ax.axes.xaxis.set_major_locator(ticker.MultipleLocator(1/show_ntotal))
    ax.axes.yaxis.set_major_locator(ticker.MultipleLocator(1/show_ptotal))
    plt.grid(True)
    return fig, ax, show_ntotal, show_ptotal
#enddef

def plotROC(fpr, tpr, plotTitle, numThresh, thresh, fancyLabel):
    ''' returns fig, ax '''

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    # ax  = plt.add_subplot(1, 1, 1, xticks=[], yticks=[])

    plt.plot(fpr, tpr, color='blue', lw=2)

    # add threshold labels and circles
    # allow up to numThresh labels per plot, or numThresh+4 in the first multiple
    if 'thresh' in locals():
        if 'fancyLabel' not in locals():
            fancyLabel = False
        addPoints(fpr, tpr, numThresh, thresh, fancyLabel)
    #endif

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plotTitle)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    ax.axes.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.axes.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    return fig, ax
#enddef

def plotSimpleROC(fpr,tpr,title):
    plt.plot(fpr, tpr)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
#enddef

def plotOpt(fpr_opt, tpr_opt, thresh_opt, maxThresh, fancyLabel):
    # plot optimal ROC points
    plt.scatter(fpr_opt, tpr_opt, s=30, marker='o', alpha=1, facecolors='w', edgecolors='r')
    for fpr, tpr, thresh in zip(fpr_opt, tpr_opt, thresh_opt):
        label, offset, fontsize = get_ROC_Curve_Label_Offset_Fontsize(fpr, tpr, thresh, maxThresh, fancyLabel)
        plt.annotate(label, (fpr, tpr), textcoords="offset points",
                     color='r', xytext=offset, ha='left', fontsize=fontsize)
    #endfor
    return
#enddef

def plotPartialArea(pfpr, ptpr, showError):
    # plot partial areas in ROC plots or concordance matrix plots

    # define lines for partial area: left line (ll), right line (rl),
    #                                bottom line (bl), top line (tl)
    SWpoint_yStripe = [pfpr[0],  0]
    NWpoint_yStripe = [pfpr[0],  1]
    NEpoint_yStripe = [pfpr[-1], 1]
    SEpoint_yStripe = [pfpr[-1], 0]
    #
    SWpoint_xStripe = [0, ptpr[0] ]
    NWpoint_xStripe = [0, ptpr[-1]]
    NEpoint_xStripe = [1, ptpr[-1]]
    SEpoint_xStripe = [1, ptpr[0] ]
    #
    # take sequences of x and y (horizontal or vertical) and plot them...
    plotLine   = lambda x, y: plt.plot(x, y, 'k--',    color=(0.5, 0.5, 0.5),   linewidth=1.5)
    plotpAUCy  = lambda x, y: plt.fill(x, y, 'xkcd:yellow', alpha=0.5,          linewidth=None)
    plotpAUCx  = lambda x, y: plt.fill(x, y, 'b',      alpha=0.4,               linewidth=None)
    plotClear  = lambda x, y: plt.fill(x, y, 'w',                               linewidth=None)
    plotpAUCxy = lambda x, y: plt.fill(x, y, 'g',      alpha=0.4,               linewidth=None)
    plotError  = lambda x, y: plt.fill(x, y, 'r',      alpha=0.25,              linewidth=None)
    plotErrorxy= lambda x, y: plt.fill(x, y, 'r',      alpha=0.35,              linewidth=None)
    #
    # plot vertical stripe lines
    plotLine([SWpoint_yStripe[0], NWpoint_yStripe[0]], [SWpoint_yStripe[1], NWpoint_yStripe[1]])
    plotLine([SEpoint_yStripe[0], NEpoint_yStripe[0]], [SEpoint_yStripe[1], NEpoint_yStripe[1]])

    # plot vertical AUCy in muted yellow
    x = pfpr + [SEpoint_yStripe[0]] + [SWpoint_yStripe[0]] + [pfpr[0]]
    y = ptpr + [SEpoint_yStripe[1]] + [SWpoint_yStripe[1]] + [ptpr[0]]
    plotpAUCy(x, y)

    # plot vertical   Error in light orange
    if showError:
       x = [NWpoint_yStripe[0]] + [NEpoint_yStripe[0]] + [NEpoint_yStripe[0]] + [NWpoint_yStripe[0]]
       y = [NWpoint_xStripe[1]] + [NEpoint_xStripe[1]] + [NEpoint_yStripe[1]] + [NWpoint_yStripe[1]]
       plotError(x, y)
    #endif

    # plot horizontal stripe lines
    plotLine([SWpoint_xStripe[0], SEpoint_xStripe[0]], [SWpoint_xStripe[1], SEpoint_xStripe[1]])
    plotLine([NWpoint_xStripe[0], NEpoint_xStripe[0]], [NWpoint_xStripe[1], NEpoint_xStripe[1]])

    # plot horizontal AUCx in slightly muted blue
    x = pfpr + [NEpoint_xStripe[0]] + [SEpoint_xStripe[0]] + [pfpr[0]]
    y = ptpr + [NEpoint_xStripe[1]] + [SEpoint_xStripe[1]] + [ptpr[0]]
    plotpAUCx(x, y)

    # plot horizontal Error in light orange
    if showError:
       x = [SWpoint_xStripe[0]] + [SWpoint_yStripe[0]] + [NWpoint_yStripe[0]] + [NWpoint_xStripe[0]]
       y = [SWpoint_xStripe[1]] + [SEpoint_xStripe[1]] + [NEpoint_xStripe[1]] + [NWpoint_xStripe[1]]
       plotError(x, y)
    #endif

    # plot overlap in AUCxy as muted green; pAUCc = pAUCx + pAUCy + pAUCxy
    x = pfpr + [SEpoint_yStripe[0]] + [pfpr[0]]
    y = ptpr + [SEpoint_xStripe[1]] + [ptpr[0]]
    plotClear(x, y)   # first clear overlap area (using white)
    plotpAUCxy(x, y)  # then plot/fill muted green
    plt.plot(pfpr, ptpr, 'b-', linewidth=2)  # replot the partial curve
    #
    # plot overlap    Error in light red
    if showError:
        x = pfpr + [NWpoint_yStripe[0]] + [pfpr[0]]
        y = ptpr + [NWpoint_xStripe[1]] + [ptpr[0]]
        plotErrorxy(x, y)
    #endif

    return
#enddef

def addPoints(fpr, tpr, numThresh, thresh, fancyLabel):
    # add threshold labels and circles
    # allow up to numThresh labels per plot, or numThresh+4 in the first multiple
    if not math.isinf(thresh[0]):
        maxThresh = thresh[0]  # if first (max) thresh is not infinite, then use it for label
    else:
        maxThresh = thresh[1]  # otherwise, use the next label which should be finite
    #endif

    stepfactor = round((len(thresh)-4) / numThresh)
    if stepfactor == 0:
        stepfactor = 1
    #endif

    for i in range(0, len(thresh), stepfactor):
        label, offset, fontsize = \
            get_ROC_Curve_Label_Offset_Fontsize(fpr[i], tpr[i], thresh[i], maxThresh, fancyLabel)
        if fancyLabel:
            plt.annotate(label, (fpr[i], tpr[i]), textcoords="offset points",
                         xytext=offset, ha='left', fontsize=fontsize)
        #endif
        plt.scatter(fpr[i], tpr[i], s=8, color='blue')
    #endfor
#enddef

def makeLabels01(labels, posclass):
    posIndex  = list(np.argwhere(np.array(labels) == posclass).flatten())
    negIndex  = list(np.argwhere(np.array(labels) != posclass).flatten())
    newlabels = labels.copy()
    for i in posIndex:
        newlabels[i] = 1
    for i in negIndex:
        newlabels[i] = 0
    return newlabels
#enddef

def rocErrorCheck(fpr,tpr,thresh,rangeEndpoints1,rangeAxis,rocRuleLeft,rocRuleRight):
    FPR, TPR, NE, SW = ('FPR', 'TPR', 'NE', 'SW')
    # error checks
    if len(thresh) != len(fpr) or len(thresh) != len(tpr):
        print("fpr, tpr and thresholds must have the same length")
    if len(thresh) < 2:
        raise ValueError('There must be at least 2 points in fpr, tpr, thresholds')
    if rangeEndpoints1[0] >= rangeEndpoints1[1] or len(rangeEndpoints1) != 2:
        raise ValueError('Improper range: wrong length or reversed order')
    if rangeEndpoints1[0] < 0 or rangeEndpoints1[0] > 1 \
            or rangeEndpoints1[1] < 0 or rangeEndpoints1[1] > 1:
        raise ValueError('Improper range: it must be in [0, 1]')
    if rocRuleLeft != NE and rocRuleLeft != SW:
        raise ValueError('Improper rocRuleLeft: it must be NE or SW')
    if rocRuleRight != NE and rocRuleRight != SW:
        raise ValueError('Improper rocRuleRight: it must be NE or SW')
    if rangeAxis != TPR and rangeAxis != FPR:
        raise ValueError('rangeAxis must be FPR or TPR')
    return
#enddef

def cSanityCheck(range):
    if range[0] < 0 or range[0] > 1 \
            or range[1] < 0 or range[1] > 1:
        raise ValueError('Improper range: it must be in [0, 1]')
    return
#enddef

def cErrorCheck(cRuleLeft, cRuleRight):
    if cRuleLeft  != "NE" and cRuleLeft  != "SW" and cRuleLeft  != "NE+1" and cRuleLeft  != "SW+1":
        raise ValueError('Improper cRuleLeft: it must be NE, NE+1, SW or SW+1')
    if cRuleRight != "NE" and cRuleRight != "SW" and cRuleRight != "NE+1" and cRuleRight != "SW+1":
        raise ValueError('Improper cRuleRight: it must be NE, NE+1, SW or SW+1')
    return
#enddef

def areEpsilonEqual(a, b, atext, btext, ep):
    ''' check equality with allowance for round-off errors up to epsilon '''
    fuzzyEQ = lambda a, b, ep: (np.abs(a - b) < ep)
    if fuzzyEQ(a, b, ep):
        print(f"PASS: {atext:12s} ({a:0.4f}) {'matches':<14s} {btext:<12s} ({b:0.4f})")
        return True
    else:
        print(f"FAIL: {atext:12s} ({a:0.4f}) {'does not match':<14s} {btext:<12s} ({b:0.4f})")
        return False
    #endif
#enddef

def Heaviside(x):
    if   x > 0:
        return 1
    elif x < 0:
        return 0
    elif x == 0:
        return 0.5
    else:
        raise ValueError('Unexpected value for x')
    #endif
#enddef

def get_plabel(fnewlabel, matchedIndices, approxIndices):
    if matchedIndices[0] == 'NA':
        print('Warning: for avgBA using an approximate left  label (no interpolation)')
        left = approxIndices[0]
    else:
        left = matchedIndices[0]
    #endif
    if matchedIndices[1] == 'NA':
        print('Warning: for avgBA using an approximate right label (no interpolation)')
        right = approxIndices[1]
    else:
        right = matchedIndices[1]
    # endif
    return fnewlabel[left:right]
#enddef

def do_pAUCc(mode,          index,     pAUCrange,
             labels,        scores,    posclass,
             fpr,           tpr,       thresh,
             fpr_opt,       tpr_opt,   thresh_opt,
             numShowThresh, testNum,   showPlot,
             showData,      showError, ep,
             rangeAxis,     useCloseRangePoint):

    ''' insert docstring here '''
    # STATUS: INCOMPLETE

    if index == 0:
        print(f'\n{index}. Whole curve baseline check\n')
        plotTitle = f'ROC curve in {mode}'
        matrixTitle = f'Concordance matrix in {mode}'

        # Show the standard ROC data
        if showData:
            print('Python\'s built-in standard ROC data')
            print(f"{'fpr':6s}, {'tpr':6s}, {'thresh':6s}")
            for x, y, t in zip(fpr, tpr, thresh):
                print(f'{x:-6.3g}, {y:-6.3g}, {t:-6.3g}')
                # print(f'{x:0.3f}, {y:0.3f}, {t:0.3f}')
            # endfor
            print(' ')
        # endif

        # Plot the standard ROC data
        if showPlot:
            pass
            # fancyLabel = False
            # # show the optional thresholds too
            # fig, ax = plotROC(fpr, tpr, 'Python\'s built-in standard ROC data', numShowThresh, thresh, fancyLabel)
            # plt.show()
            # if index > 0:
            #     modeShort = mode[:-3]  # training -> train, testing -> test
            #     fig.savefig(f'output/ROC{modeShort}{testNum}-{index}std.png')
            # #endif
            # plt.close(fig)
        # endif
    else:
        print(f'\n{index}. Partial curve {index}\n')
        plotTitle   = f'Partial ROC curve {index} in {mode}'
        matrixTitle = f'Concordance matrix part {index} in {mode}'
    #endif

    # Get the "full" ROC using our own ROC function (sklearn's is
    # insufficient since it does not include ties).  The full ROC
    # is necessary in 2 scenarios:
    #   1) for the partial c statistic
    #   2) to show all thresholds in an ROC for decision-making
    #   NOTE: It is NOT necessary for the concordant partial AUC
    ffpr, ftpr, fthresh, fnewlabel = getFullROC(labels, scores, posclass)
    # data are returned, sorted in ascending order by fpr, then tpr within fpr

    # Show the full ROC data
    if showData:
        print(f"{'ffpr':6s}, {'ftpr':6s}, {'fthresh':7s}")
        for x, y, t in zip(ffpr, ftpr, fthresh):
            print(f'{x:-6.3g}, {y:-6.3g}, {t:-6.3g}')
        #endfor
        print(' ')
    #endif

    # used by plotOpt later
    if not math.isinf(fthresh[0]):
        maxThresh = fthresh[0]  # if first (max) thresh is not infinite, then use it for label
    else:
        maxThresh = fthresh[1]  # otherwise, use the next label which should be finite
    # endif

    # plot full ROC data for whole curve with thresholds labeled and the Metz optimal ROC point(s) indicated
    if showPlot and index == 0:
        fancyLabel = True
        fig, ax    = plotROC(ffpr, ftpr, plotTitle, numShowThresh, fthresh, fancyLabel)
        plotOpt(fpr_opt, tpr_opt, thresh_opt, maxThresh, fancyLabel)  # add the optimal ROC points
        plt.show()
        modeShort = mode[:-3]  # training -> train, testing -> test
        fig.savefig(f'output/ROC_{modeShort}_{testNum}-{index}.png')
    # endif

    # The common and simplistic view of AUC as a vertical integral is generalized
    # by our "concordant partial AUC" pAUCc as a vertical and horizontal integral.
    # For a partial curve this distinction matters because the "partial AUC" is
    # improperly skewed to consider positives more than negatives. Our "concordant
    # partial AUC" is the proper generalization of AUC, with a dual positive and negative
    # (vertical and horizontal) perspective. Note that the dual perspective has one
    # additional complexity:
    #
    #    If a partial range in FPR ends on the vertical part of a staircase step,
    #    then we need to specify if it ends at the bottom or top of the step. Or
    #    for a partial range in TPR the ambiguity is horizontal: left or right.
    #    We resolve any ambiguities that arise by using either the most
    #    SouthWest (SW) point or the most NorthEast (NE) point, for each end of
    #    the partial range:
    #
    if index == 0 or index == 1:   # index 1,2,... for partial curves, 0 for whole
        rocRuleLeft  = 'SW'
        rocRuleRight = 'NE'
    else:
        rocRuleLeft  = 'NE'
        rocRuleRight = 'NE'
    #endif

    # swith pAUCrange to the closest points, if applicable
    if useCloseRangePoint:
        if rangeAxis == 'FPR':
            rstat = ffpr
            ostat = ftpr
        else:
            rstat = ftpr
            ostat = ffpr
        #endif
        for i in [0, 1]:
            ix, ixL, ixR = get_Match_or_Interpolation_Points(rstat, pAUCrange[i])
            if len(ix) == 0:  # if no exact match, then use closest point
                _, __, perCentFromLeft = interpolateROC(rstat, ostat, fthresh, ixL, ixR, pAUCrange[i])
                if perCentFromLeft > 0.5:
                    pAUCrange[i] = rstat[ixR]
                else:
                    pAUCrange[i] = rstat[ixL]
                #endif
            #endif
        #endfor
    #endif

    # get range and rangeIndices for pAUCc, using full ROC data
    pfpr, ptpr, otherpAUCrange, trange, matchedIndices, approxIndices \
        = getRange_pAUCc(ffpr, ftpr, fthresh, rangeAxis, pAUCrange, rocRuleLeft, rocRuleRight)
    if rangeAxis == 'FPR':
        xrange = pAUCrange
        yrange = otherpAUCrange
    else:
        yrange = pAUCrange
        xrange = otherpAUCrange
    #endif
    #xrange = pAUCrange
    #pfpr, ptpr, yrange, trange, matchedIndices, approxIndices \
    #    = getRange_pAUCc(ffpr, ftpr, fthresh, 'FPR', xrange, rocRuleLeft, rocRuleRight)

    # Show the partial ROC data
    if showData:
        print(f"{'pfpr':6s}, {'ptpr':6s}")
        for x, y in zip(pfpr, ptpr):
            print(f'{x:-6.3g}, {y:-6.3g}')
        #endfor
        print(' ')
    #endif

    # get range and rangeIndices for cDelta, using full ROC data
    negIndexC, posIndexC, negScores, posScores, negWeights, posWeights \
        = getRange_cDelta(ffpr, ftpr, fthresh, scores, labels, posclass, xrange, yrange)

    # plot full ROC data for whole and partial curve
    # with thresholds labeled and the Metz optimal ROC point(s) indicated
    if showPlot and index > 0:
        fancyLabel = False  # we apply points and score labels later, not now
        fig, ax    = plotROC(ffpr, ftpr, plotTitle, numShowThresh, fthresh, fancyLabel)
        plotPartialArea(pfpr, ptpr, showError)  # add fills for partial areas (clobbers points, score labels)
        fancyLabel = True            # now we apply the points and score labels
        addPoints(ffpr, ftpr, numShowThresh, fthresh, fancyLabel)     # add points and score labels
        plotOpt(fpr_opt, tpr_opt, thresh_opt, maxThresh, fancyLabel)  # add the optimal ROC points
        plt.show()
        modeShort = mode[:-3]  # training -> train, testing -> test
        fig.savefig(f'output/ROC_{modeShort}_{testNum}-{index}.png')
    # endif

    # plot concordance matrix
    if showPlot:
        maxInstancesPerAxis = 20
        fig, ax, show_ntotal, show_ptotal = \
            plotConcordanceMatrix(fpr, tpr, negScores, posScores, matrixTitle, maxInstancesPerAxis)

        if index > 0:
            plotPartialArea(pfpr, ptpr, showError)  # add fills for partial areas (clobbers points, score labels)
        #endif

        # add points, score labels, optimal ROC points
        fancyLabel = True
        addPoints(ffpr, ftpr, numShowThresh, fthresh, fancyLabel)
        plotOpt(fpr_opt, tpr_opt, thresh_opt, maxThresh, fancyLabel)
        plt.show()
        modeShort = mode[:-3]  # training -> train, testing -> test
        fig.savefig(f'output/cMatrix_{modeShort}_{testNum}-{index}.png')
    #endif

    print(f"{'FPR':12s} = [{xrange[0]:0.3f} {xrange[1]:0.3f}]")
    print(f"{'TPR':12s} = [{yrange[0]:0.3f} {yrange[1]:0.3f}]")
    if np.isinf(trange[0]):
        print(f"{'Thresholds':12s} = [{'inf':<5s} {trange[1]:0.3f}]")
    else:
        print(f"{'Thresholds':12s} = [{trange[0]:0.3f} {trange[1]:0.3f}]")
    #endif

    print(f"{' ':29s} {'fpr':<6s} {'tpr':<6s} {'thresh':<6s}      {'fpr':<6s} {'tpr':<6s} {'thresh':<6s}")
    if np.isinf(trange[0]):
        print(f"{'rocLeftRight':<12s} = [{rocRuleLeft:<4s} {rocRuleRight:<4s}]   "
              f"({xrange[0]:0.4f},{yrange[0]:0.4f},{'inf':<6s}) to "
              f"({xrange[1]:0.4f},{yrange[1]:0.4f},{trange[1]:0.4f})")
    else:
        print(f"{'rocLeftRight':<12s} = [{rocRuleLeft:<4s} {rocRuleRight:<4s}]   "
              f"({xrange[0]:0.4f},{yrange[0]:0.4f},{trange[0]:0.4f}) to "
              f"({xrange[1]:0.4f},{yrange[1]:0.4f},{trange[1]:0.4f})")
    #endif
    print(' ')

    p1 = int(posIndexC[0]); p2 = int(posIndexC[1])
    n1 = int(negIndexC[0]); n2 = int(negIndexC[1])
    print(f"{'positive indices':16s} = [{p1:2d} {p2:2d}]")
    print(f"{'negative indices':16s} = [{n1:2d} {n2:2d}]")

    print(f"{' ':29s} {'nScore':<6s}  {'ix':<4s},  {'pScore':<6s}  {'ix':<4s}       "
                    f"{'nScore':<6s}  {'ix':<4s},  {'pScore':<6s}  {'ix':<4s}")
    print(f"{'rocLeftRight':<12s} = [{rocRuleLeft:<4s} {rocRuleRight:<4s}]   "
          f"({negScores[n1]:0.4f} ({n1:4d}), {posScores[p1]:0.4f} ({p1:4d})) to "
          f"({negScores[n2]:0.4f} ({n2:4d}), {posScores[p2]:0.4f} ({p2:4d}))")
    print(f"with weights: {' ':16s}"
          f"{float(negWeights[n1]):0.4f}{' ':9s}{float(posWeights[p1]):0.4f}{' ':13s}"
          f"{float(negWeights[n2]):0.4f}{' ':9s}{float(posWeights[p2]):0.4f}")
    print(' ')

    extras_dict = dict()
    if index == 0:
        c   = c_statistic(posScores, negScores)
        print(f"{'c':12s} = {c:0.4f}")
        print(' ')
        extras_dict.update({'c': c})
    #endif

    cDelta, cDeltan, cLocal \
        = partial_c_statistic(posScores, negScores, posWeights, negWeights)
    print(f"{'cDelta':12s} = {cDelta:0.4f}")
    print(f"{'cDeltan':12s} = {cDeltan:0.4f}")
    print(f"{'cLocal':12s} = {cLocal:0.4f}")
    print(' ')

    if index == 0:
        AUC = metrics.auc(fpr, tpr)
        print(f"{'AUC':12s} = {AUC:0.4f}")
        print(' ')
        extras_dict.update({'AUC': AUC})
    #endif

    # make a partial series from ffpr and ftpr and add interpolated endpoints as needed

    pAUCc, pAUC, pAUCx, pAUCcn, pAUCn, pAUCxn \
        = concordant_partial_AUC(pfpr, ptpr)
    print(f"{'pAUC':12s} = {pAUC:0.4f}")
    print(f"{'pAUCx':12s} = {pAUCx:0.4f}")
    print(f"{'pAUCc':12s} = {pAUCc:0.4f}")
    print(f"{'pAUCn':12s} = {pAUCn:0.4f}")
    print(f"{'pAUCxn':12s} = {pAUCxn:0.4f}")
    print(f"{'pAUCcn':12s} = {pAUCcn:0.4f}")
    print(' ')

    sPA = standardized_partial_area_proxy(pfpr, ptpr)
    print(f"{'sPA':12s} = {sPA:0.4f}")
    extras_dict.update({'sPA': sPA})


    plabel = get_plabel(fnewlabel, matchedIndices, approxIndices)
    avgSens, avgSpec  = average_measures_discrete(pfpr, ptpr, plabel)
    print(f"{'avgSens':12s} = {avgSens:0.4f}")
    print(f"{'avgSpec':12s} = {avgSpec:0.4f}")
    extras_dict.update({'avgSens': avgSens})
    extras_dict.update({'avgSpec': avgSpec})

    if xrange[1] == 1:
        PAI = partial_area_index_proxy(pfpr, ptpr)
        print(f"{'PAI':12s} = {PAI:0.4f}  only applies to full or last range")
        extras_dict.update({'PAI': PAI})
    #endif
    print(' ')

    # check for expected equalities
    pass1 = areEpsilonEqual(cDelta,  pAUCc,  'cDelta',  'pAUCc',  ep)
    pass2 = areEpsilonEqual(cDeltan, pAUCcn, 'cDeltan', 'pAUCcn', ep)
    print(' ')

    return (pass1 and pass2), cDelta, pAUCc, pAUC, pAUCx, cDeltan, pAUCcn, pAUCn, pAUCxn, extras_dict
#enddef
