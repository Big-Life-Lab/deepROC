#!/usr/bin/env python
# -*- coding: latin-1 -*-
# DeepROCFunctions.py
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

# imports are locally defined in each function

def Heaviside(x):  # handles ties properly
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

def concordant_partial_AUC(pfpr, ptpr, quiet):
    import numpy as np
    ''' Computes the concordant partial area under the curve and related parts, \n
        given arrays of "partial fpr" and "partial tpr" values.  These arrays only \n
        contain points on the partial curve and the trapezoidal rule is used to compute \n
        areas with these points. \n
    \n
    AUC_i:      the concordant partial area under the curve \n
    pAUC:       the (vertical) partial area under the curve \n
    pAUCx:      the horizontal partial area under the curve \n
    \n
    AUCn_i:     the concordant partial area under the curve normalized \n
    pAUCn:      the (vertical) partial area under the curve normalized \n
    pAUCxn:     the horizontal partial area under the curve normalized \n
    '''

    # xgroups [a,b]
    a    = float(pfpr[0])
    b    = float(pfpr[-1])
    delx = b - a
    vertical_stripe_area = (1 * delx)

    # ygroups [f,g]
    f    = float(ptpr[0])
    g    = float(ptpr[-1])
    dely = g - f
    horizontal_stripe_area = (dely * 1)

    if delx == 0:
        if not quiet:
            print("Warning: For pAUC and pAUCc the width (delx) of the vertical column is zero.")
        #endif
        # for a region with no width or area...
        pAUC  = 0  # contribution to AUC from pAUC is zero
        pAUCn = 0  # contribution to pAUCn (= avgSens) is zero
    else:
        # Compute the partial AUC mathematically defined in (Dodd and Pepe, 2003) and conceptually
        # defined in (McClish, 1989; Thompson and Zucchini, 1989). Use the trapezoidal rule to 
        # compute the integral.
        pAUC  = np.trapz(ptpr, pfpr)  # trapz is y,x
        pAUCn = pAUC / vertical_stripe_area
    #endif

    if dely == 0:
        if not quiet:
            print("Warning: For pAUCx and pAUCc the height (dely) of the horizontal stripe is zero.")
        #endif
        # for a region with no width or area...
        pAUCx  = 0  # contribution to AUC from pAUCx is zero
        pAUCxn = 0  # contribution to pAUCxn (= avgSpec) is zero
    else:
        # Compute the horizontal partial AUC (pAUCx) defined in (Carrington et al, 2020) and as
        # suggested by (Walter, 2005) and similar to the partial area index (PAI) (Jiang et al, ?)
        # although PAI has a fixed right boundary instead and a slightly different mathematical
        # definition.
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

    AUC_i  = (1/2)*pAUCx + (1/2)*pAUC  # the complexity is in the derivation, meaning/generalization
                                       #   and relation to the C and partial C statistic, not the
                                       #   formula which looks like a simple average

    total_for_norm = vertical_stripe_area + horizontal_stripe_area
    if total_for_norm == 0:
        AUCn_i = 0
        print('Warning: Zero length partial curve specified.')
    else:
        AUCn_i = (pAUCx + pAUC) / total_for_norm
    #endif

    measure_dict = dict(AUC_i=AUC_i, pAUC=pAUC, pAUCx=pAUCx, AUCn_i=AUCn_i, pAUCn=pAUCn, pAUCxn=pAUCxn)
    return measure_dict
    # return AUC_i, pAUC, pAUCx, AUCn_i, pAUCn, pAUCxn
#enddef

def partial_C_statistic_simple(posScores, negScores, posWeights, negWeights):
    ''' partial_C_simple computes the partial C statistic with no interpolation \n
        within an instance and related measures.  This occurs in two situations: \n
        groups by instances, or when the setting: groupByClosestInstance=True is used.\n
        \n
        It takes a vector of scores for actual positives and a vector of scores for \n
        actual negatives in the ROC data and corresponding weight vectors with      \n
        binary elements {0, 1} that indicate which instances are in the group being \n
        measured (1) or not (0). Notably, the group does not have to be contiguous  \n
        in ROC space, and generally or usually is not.'''

    P      = len(posScores)
    N      = len(negScores)
    partP  = float(sum(posWeights))
    partN  = float(sum(negWeights))

    # measure the contribution to the partial C statistic from horizontal slices that 
    # correspond to positives in the group. We know from the concordance matrix
    # that positives are along the y-axis.    
    count_x = 0  # cp
    for j in range(0, P):  # horizontal slices are for positives along the y axis
        if posWeights[j] == 0:  # skip positives not in the group
            continue
        # for a positive within the group
        for k in range(0, N): 
            # count how many cells along the horizontal slice are ranked correctly
            count_x = count_x + Heaviside(posScores[j] - negScores[k])
        # endfor
    # endfor

    # measure the contribution to the partial C statistic from vertical slices that 
    # correspond to negatives in the group. We know from the concordance matrix
    # that negatives are along the x-axis.    
    count_y = 0  # cn
    for k in range(0, N):  # vertical slices are for negatives along the x axis
        if negWeights[k] == 0:  # skip negatives not in the group
            continue
        # for a negative within the group
        for j in range(0, P): 
            # count how many cells along the vertical slice are ranked correctly
            count_y = count_y + Heaviside(posScores[j] - negScores[k])
        # endfor
    # endfor
    
    whole_area     = N * P                            # all cells
    C_area_x       = count_x / whole_area             # cp1
    C_area_y       = count_y / whole_area             # cn1
    partial_C      = (1/2)*C_area_x + (1/2)*C_area_y  # partial_C is an area
    
    horizontal_stripe_area = partP * N  # positives in group by full horizontal range 
    vertical_stripe_area   = partN * P  # negatives in group by full vertical range 
    C_avgSpec      = count_x   / horizontal_stripe_area  # avg sensitivity
    C_avgSens      = count_y   / vertical_stripe_area    # avg specificity
    
    # partial C normalized, is balanced average accuracy
    partial_Cn   = partial_C / ((1/2)*(horizontal_stripe_area + vertical_stripe_area))

    measure_dict = dict(C_i=partial_C,   C_area_y=C_area_y,    C_area_x=C_area_x,
                        Cn_i=partial_Cn, Cn_avgSens=C_avgSens, Cn_avgSpec=C_avgSpec)
    return measure_dict
    #return partial_C, C_area_y,  C_area_x, partial_Cn, C_avgSens, C_avgSpec
#enddef

def partial_C_statistic(posScores, negScores, posWeights, negWeights, yrange, xrange):
    ''' partial_c_statistic computes the cStatistic given a vector of scores for actual
        positives and a vector of scores for actual negatives in the ROC data and
        corresponding weight vectors with elements in [0,1] that indicate which instances
        are in the partial range and how much weight for each instance.  That is,
        boundary instances may have partial weight, interior instances will have weight 1
        and exterior instances will have weight 0'''

    # NOTE: THERE IS ONE CASE NOT WRITTEN/HANDLED YET (test dataset 12) WHEN THE TOP AND
    # BOTTOM (OR LEFT AND RIGHT) BOUNDS OF THE PARTIAL CURVE ARE WITHIN THE SAME CELL,
    # THE LOGIC DOES NOT YET HANDLE THIS PROPERLY.  TIME IS THE ONLY LIMITATION...

    P      = len(posScores)
    N      = len(negScores)

    partP  = float(sum(posWeights))
    partN  = float(sum(negWeights))

    # for the case of a flat vertical range, or a flat horizontal range, the scores and
    # weights do not tell us enough to compute the partial c statistic, so we need
    # xrange and yrange which are indices including decimals for interpolation
    if partP == 0:  # if range in horizontal, no vertical (no positives)
        posIndexC_0_1   = int(yrange[0])      # round down to int, for index
        remaining_posWeight = float(yrange[0]) - posIndexC_0_1
    #endif
    if partN == 0:  # if range in vertical, no horizontal (no negatives)
        negIndexC_0_1   = int(N-float(xrange[0]))  # round down to int, for index
        remaining_negWeight = N-float(xrange[0]) - negIndexC_0_1
    #endif

    leftIndex = -1  # default value to detect
    for k in range(0, N):
        if negWeights[k] > 0:
            leftIndex = k
            break
        #endif
    #endfor
    rightIndex = -1  # default value to detect
    for k in range(N-1, -1, -1):
        if negWeights[k] > 0:
            rightIndex = k
            break
        #endif
    #endfor
    bottomIndex = -1  # default value to detect
    for j in range(0, P):
        if posWeights[j] > 0:
            bottomIndex = j
            break
        # endif
    # endfor
    topIndex    = -1  # default value to detect
    for j in range(P-1, -1, -1):
        if posWeights[j] > 0:
            topIndex = j
            break
        #endif
    #endfor

    # compute positive count cp (in the horizontal stripe) based on:
    #   the subset of positives in the partial curve (non-zero posWeights)
    #   the negatives to the right of the partial curve (negWeights with indices
    #     higher than or equal to the horizontally leftmost non-zero negWeights index)
    if topIndex == -1 or bottomIndex == -1:  # no horizontal stripe
        cp = 0
    elif partN == 0:  # horizontal stripe but no vertical stripe and no negatives
        cp = 0
        for k in range(negIndexC_0_1, N):  # enforce under curve positives (no partial index)
            for j in range(0, P):
                if posWeights[j] == 0:     # enforce subset of positives
                    continue
                #endif
                # outside subset of negatives use one weight, no h
                cp = cp + float(posWeights[j])
            # endfor
        # endfor
        cp = cp + (remaining_negWeight * partP)  # add partial negative index/weight
    else:
        cp = 0
        for k in range(leftIndex, N):   # enforce right of curve negatives
            for j in range(0, P):
                if posWeights[j] == 0:  # enforce subset of positives
                    continue
                #endif
                # h==1 correct rank, h==0.5 tie, h==0 wrong rank
                h    = Heaviside(float(posScores[j]) - float(negScores[k]))
                posW = float(posWeights[j])
                negW = float(negWeights[k])
                if negW > 0:  # within subset of negatives use both weights
                    cp = cp + posW * negW * h
                else:  # outside subset of negatives use one weight, no h
                    cp = cp + posW
                if k == rightIndex and 0 < negW < 1:  # in this case
                    cp = cp + posW * (1 - negW)      # also add this part
                #endif
            #endfor
        #endfor
    #endif

    # compute negative count cn (in the vertical stripe) based on:
    #   the subset of negatives in the partial curve (non-zero negWeights)
    #   the positives under the partial curve (posWeights with indices
    #     lower than or equal to the vertically topmost non-zero posWeights index)
    #     note: posWeights is directional, so if at the bottom of a range,
    #           it must be reversed: 1-posWeights
    if leftIndex == -1 or rightIndex == -1:  # no vertical stripe
        cn = 0
    elif partP == 0:  # vertical stripe but no horizontal stripe and no positives
        cn = 0
        for j in range(0, posIndexC_0_1+1):  # enforce under curve positives, no partial index
            for k in range(0, N):
                if negWeights[k] == 0:           # enforce subset of negatives
                    continue
                # outside subset of positives use one weight, no h
                cn = cn + float(negWeights[k])
            # endfor
        # endfor
        cn = cn + (remaining_posWeight * partN)  # add partial positive index/weight
    else:  # vertical stripe normal case
        cn = 0
        for j in range(0, topIndex+1):  # enforce under curve positives
            for k in range(0, N):
                if negWeights[k] == 0:  # enforce subset of negatives
                    continue
                # h==1 correct rank, h==0.5 tie, h==0 wrong rank
                h    = Heaviside(float(posScores[j]) - float(negScores[k]))
                posW = float(posWeights[j])
                negW = float(negWeights[k])
                if posW > 0:  # within subset of positives use both weights and h
                    cn = cn + negW * posW * h
                else:  # outside subset of positives use one weight, no h
                    cn = cn + negW
                if j == bottomIndex and 0 < posW < 1:  # in this case
                    cn = cn + negW * (1 - posW)        # also add this part
                #endif
            #endfor
        #endfor
    #endif

    cLocal = 0
    for j in range(0, P):
        for k in range(0, N):
            h      = Heaviside(float(posScores[j]) - float(negScores[k]))
            cLocal = cLocal + float(posWeights[j]) * \
                     float(negWeights[k]) * h
        #endfor
    #endfor

    whole_area             = N * P
    cDelta      = (1/2)*(cp/whole_area) + (1/2)*(cn/whole_area)
    horizontal_stripe_area = N * partP
    vertical_stripe_area   = P * partN
    # OLD overall normalization:
    if   partP == 0 and partN == 0:
        cDeltan = 0
    else:
        cDeltan= (cDelta * (2*whole_area)) / (horizontal_stripe_area + vertical_stripe_area)
    #endif

    # NEW piece-wise normalization:
    if   partP == 0 and partN == 0:
        cn_normalized = 0
        cp_normalized = 0
    elif partP == 0:
        cn_normalized = cn/vertical_stripe_area
        cp_normalized = 0
    elif partN == 0:
        cn_normalized = 0
        cp_normalized = cp/horizontal_stripe_area
    else:
        cn_normalized = cn/vertical_stripe_area
        cp_normalized = cp/horizontal_stripe_area
    #endif
    #cDeltan = (1/2) * cp_normalized + (1/2) * cn_normalized

    local_area = partP * partN
    if local_area == 0:
        cLocal = 0
    else:
        cLocal = cLocal / local_area
    #endif

    measure_dict = dict(C_i=cDelta,     C_area_y=cn/whole_area,   C_area_x=cp/whole_area,
                        Cn_i=cDeltan,   Cn_avgSens=cn_normalized, Cn_avgSpec=cp_normalized,
                        C_local=cLocal, Uy_count=cn,              Ux_count=cp)
    return measure_dict, whole_area, horizontal_stripe_area, vertical_stripe_area

    #return cDelta, cDeltan, cLocal, cp/whole_area, cn/whole_area, cp_normalized, cn_normalized, cp, cn, \
    #    whole_area, horizontal_stripe_area, vertical_stripe_area
#enddef

def discrete_partial_roc_measures(partial_fpr, partial_tpr, n_negatives, n_positives, populationPrevalence):
    '''Compute the discrete partial ROC measures using "full ROC" data (see getFullROC): \n
       average sensitivity (average recall), average specificity, balanced average accuracy \n
       (equal to the partial C statistic), average positive predictive value (average precision), \n
       average negative predictive value, average likelihood ratio positive, average likelihood
       ratio negative, average odds ratio, average accuracy, average balanced accuracy.\n '''
    import numpy as np

    pi_neg        = n_negatives / (n_positives + n_negatives)
    pi_pos        = n_positives / (n_positives + n_negatives)
    pi_neg_pop    = 1 - populationPrevalence
    pi_pos_pop    =     populationPrevalence
    sumSensArea, sumSpecArea, sumPPVArea,  sumNPVArea, sumLRpArea      = [0, 0, 0, 0, 0]
    sumLRnArea,  sumORArea,   sum_BA_Area, sum_A_Area, delx,      dely = [0, 0, 0, 0, 0, 0]
    lastfpr       = float(partial_fpr[0])
    lasttpr       = float(partial_tpr[0])

    # All computations, start from the bottom left of the ROC plot
    # omit the first point in the region, it has no area/weight
    for fpr, tpr in zip(partial_fpr[1:], partial_tpr[1:]):
        ldely       =  tpr - lasttpr
        ldelx       =  fpr - lastfpr

        # temporary variables
        avgtpr      = (tpr + lasttpr) / 2
        avgfpr      = (fpr + lastfpr) / 2

        # for bAvgA, avgSens, avgSpec
        sumSensArea = sumSensArea +  (ldelx  * avgtpr)
        sumSpecArea = sumSpecArea +  (ldely  * (1-avgfpr))

        # for avgBA, avgA, avgPPV, avgNPV, avgLRp, avgLRn -- all using ldelw=(ldelx + ldely) an L1 distance
        sum_BA_Area = sum_BA_Area + ((avgtpr + (1-avgfpr))/2)                    * (ldelx + ldely)
        sum_A_Area  = sum_A_Area  + (pi_pos * avgtpr     + pi_neg * (1-avgfpr))  * (ldelx + ldely)

        PPV_denominator                =  pi_pos_pop * avgtpr     + pi_neg_pop * avgfpr
        NPV_denominator                =  pi_neg_pop * (1-avgfpr) + pi_pos_pop * (1-avgtpr)
        if PPV_denominator != 0:
            sumPPVArea  = sumPPVArea  + ((pi_pos_pop * avgtpr) / PPV_denominator)    * (ldelx + ldely)
        if NPV_denominator != 0:
            sumNPVArea  = sumNPVArea  + ((pi_neg_pop * (1-avgfpr)) / NPV_denominator)* (ldelx + ldely)
        if avgfpr == 0:
            sumLRpArea  = np.inf
        else:
            sumLRpArea  = sumLRpArea  + (avgtpr / avgfpr)
        if (1-avgfpr) == 0:
            sumLRnArea  = np.inf
        else:
            sumLRnArea  = sumLRnArea  + ((1-avgtpr) / (1-avgfpr))

        if avgfpr == 0 or (1-avgtpr) == 0:
            sumORArea   = np.inf
        elif (1-avgfpr) == 0:
            sumORArea   = sumORArea + 0
        else:
            sumORArea   = sumORArea   + ( (avgtpr / avgfpr) / ((1-avgtpr) / (1-avgfpr)) )

        delx    = delx + ldelx
        dely    = dely + ldely
        lastfpr = fpr
        lasttpr = tpr
    #endfor
    if delx > 0:
        avgSens = (1/delx) * sumSensArea
    else:
        # avgSens is usually defined as the average height for the partial area in the range of delx.
        # When delx==0 there is no "area"
        avgSens  = 0
    #endif
    if dely > 0:
        avgSpec = (1/dely) * sumSpecArea
    else:
        # avgSpec is usually defined as the average width (from right) for the partial area in the range
        # of dely.  When dely==0 there is no "area"
        avgSpec = 0
    #endif

    sumdel  = delx + dely
    if sumdel == 0:
        bAvgA, avgBA, avgA, avgPPV, avgNPV, avgLRp, avgLRn, avgOR = [0, 0, 0, 0, 0, 1, 1, 1]
    else:
        bAvgA   = (delx/sumdel) * avgSens + (dely/sumdel) * avgSpec
        avgBA   = (1/sumdel)  * sum_BA_Area
        avgA    = (1/sumdel)  * sum_A_Area
        avgPPV  = (1/sumdel)  * sumPPVArea
        avgNPV  = (1/sumdel)  * sumNPVArea
        avgLRp  = (1/sumdel)  * sumLRpArea
        avgLRn  = (1/sumdel)  * sumLRnArea
        avgOR   = (1/sumdel)  * sumORArea
    #endif

    measures_dict = { 'avgSens': avgSens,  'avgSpec': avgSpec,
                      'avgPPV':   avgPPV,  'avgNPV':   avgNPV,
                      'avgLRp':   avgLRp,  'avgLRn':   avgLRn,
                      'bAvgA':     bAvgA,  'avgOR':    avgOR,
                      'avgA':       avgA,  'avgBA':    avgBA}
    return measures_dict
#enddef

def partial_area_index_proxy(pfpr, ptpr, quiet):
    import numpy as np
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
        if not quiet:
            print(" Warning: For PAI the height (dely) of the horizontal stripe is zero.")
        #endif
        #PAI = (1-float(pfpr[0]) + 0)/2  # average specificity on horizontal line
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

def areEpsilonEqual(a, b, atext, btext, ep, quiet):
    import numpy as np
    ''' check equality with allowance for round-off errors up to epsilon '''
    if     np.isnan(a) and np.isnan(b):
        return True
    else:
        if np.isnan(a) or  np.isnan(b):
            return False
        #endif
    #endif
    fuzzyEQ = lambda a, b, ep: (np.abs(a - b) < ep)
    if fuzzyEQ(a, b, ep):
        if not quiet:
            print(f"PASS: {atext:12s} ({a:0.4f}) {'matches':<14s} {btext:<12s} ({b:0.4f})")
        #endif
        return True
    else:
        if not quiet:
            print(f"FAIL: {atext:12s} ({a:0.4f}) {'does not match':<14s} {btext:<12s} ({b:0.4f})")
        #endif
        return False
    #endif
#enddef

def rocErrorCheck(fpr,tpr,thresh,rangeEndpoints1,rangeAxis,rocRuleLeft,rocRuleRight):
    FPR, TPR, NE, SW = ('FPR', 'TPR', 'NE', 'SW')
    # error checks
    if thresh is not None:
        if len(thresh) != len(fpr) or len(thresh) != len(tpr):
            print("fpr, tpr and thresholds must have the same length")
        if len(thresh) < 2:
            raise ValueError('There must be at least 2 points in fpr, tpr, thresholds')
    #endif
    if rangeEndpoints1[0] >= rangeEndpoints1[1] or len(rangeEndpoints1) != 2:
        pass
        # this is not necessarily an error, allow it instead of raising an error
        # raise ValueError(f'Improper range: wrong length or reversed order {rangeEndpoints1}')
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

def showCmeasures(i, C_i, C_area_x, C_area_y, Cn_i, Cn_avgSpec, Cn_avgSens,
                  Ux_count, Uy_count, whole_area, vertical_stripe_area, horizontal_stripe_area):
    delx = vertical_stripe_area
    dely = horizontal_stripe_area

    #print('Partial U statistics:')
    #print(f"Uy_{i}{'':12s} = {Uy_count:0.2f}")
    #print(f"Ux_{i}{'':12s} = {Ux_count:0.2f}")
    #print('')
    #print(f"C_{i}{'   (~AUC_i) ':13s} = {C_i:0.4f} = 1/2 Cy_{i} + 1/2 Cx_{i}")
    #print(f"Cy_{i}{'  (~pAUC)  ':12s} = {C_area_y:0.4f} = Uy_{i}/(count_x * count_y) = {Uy_count:0.2f}/{whole_area:0.2f}")
    #print(f"Cx_{i}{'  (~pAUCx) ':12s} = {C_area_x:0.4f} = Ux_{i}/(count_x * count_y) = {Ux_count:0.2f}/{whole_area:0.2f}")
    #print(f"Cn_{i}{'  (~AUCn_i)':12s} = {Cn_i:0.4f} = delx/(delx + dely) * Cny_{i} + dely/(delx + dely) * Cnx_{i}")
    #print(f"Cny_{i}{' (~pAUCn) ':11s} = {Cn_avgSens:0.4f} = Uy_{i}/del_count_x = {Uy_count:0.2f}/{vertical_stripe_area:0.2f}")
    #print(f"Cnx_{i}{' (~pAUCxn)':11s} = {Cn_avgSpec:0.4f} = Ux_{i}/del_count_y = {Ux_count:0.2f}/{horizontal_stripe_area:0.2f}")

    print(f"C_{i}{'   (~AUC_i) ':13s} = {C_i:0.4f} (Partial C statistic)")
    print(f"Cy_{i}{'  (~pAUC)  ':12s} = {C_area_y:0.4f} = {Uy_count:0.2f}/{whole_area:0.2f}")
    print(f"Cx_{i}{'  (~pAUCx) ':12s} = {C_area_x:0.4f} = {Ux_count:0.2f}/{whole_area:0.2f}")
    print('')
    print(f"Cn_{i}{'  (~AUCn_i)':12s} = {Cn_i:0.4f} (Normalized partial C statistic)")
    print(f"Cny_{i}{' (~pAUCn) ':11s} = {Cn_avgSens:0.4f} = {Uy_count:0.2f}/{vertical_stripe_area:0.2f}")
    print(f"Cnx_{i}{' (~pAUCxn)':11s} = {Cn_avgSpec:0.4f} = {Ux_count:0.2f}/{horizontal_stripe_area:0.2f}")
    print(' ')
    return
#enddef

def showWholeAUCmeasures(AUC, AUC_full, AUC_macro, AUC_micro, AUPRC):
    print(f"{'AUC':16s} = {AUC:0.4f}")
    print(f"{'AUC_full':16s} = {AUC_full:0.4f}")
    print(f"{'AUC_macro':16s} = {AUC_macro:0.4f}")
    print(f"{'AUC_micro':16s} = {AUC_micro:0.4f}")
    print(f"{'AUPRC':16s} = {AUPRC:0.4f}")
    print(' ')
    return
# enddef

def showAUCmeasures(i, AUC_i, pAUC, pAUCx, AUCn_i, pAUCn, pAUCxn):
    print(f"AUC_{i}{' ':11s} = {AUC_i:0.4f} (Concordant Partial AUC)")
    print(f"pAUC_{i}{' ':10s} = {pAUC:0.4f}")
    print(f"pAUCx_{i}{' ':9s} = {pAUCx:0.4f}")
    print(' ')
    print(f"AUCn_{i}{' ':10s} = {AUCn_i:0.4f} (Normalized concordant partial AUC)")
    print(f"pAUCn_{i}{' ':9s} = {pAUCn:0.4f}")
    print(f"pAUCxn_{i}{' ':8s} = {pAUCxn:0.4f}")
    print(' ')
    return
#enddef

def sortScoresAndLabels3(scores, newlabels, slopeFactors):
    import numpy as np
    # 3 refers to sorting a 3-variable tuple
    # sort in descending order by scores, then by newlabels (within tied scores)
    # (we must do this ascending, and then reverse it)
    # the following assumes labels are int
    dtype2     = [('scores', float), ('newlabels', int), ('slopeFactors', float)]
    rocTuples2 = list(zip(scores, newlabels, slopeFactors))
    rocArray2  = np.array(rocTuples2, dtype=dtype2)
    temp2      = np.sort(rocArray2, order=['scores', 'newlabels'])
    final2     = temp2[::-1]  # reverse the order of elements
    # put the sorted data back into the original list variables
    scores, newlabels, slopeFactors = zip(*final2)  # zip returns (immutable) tuples
    # we could convert these to lists, but we don't need to: scores = list(scores)
    return scores, newlabels, slopeFactors
#enddef

def sortScoresAndLabels4(scores, newlabels, labels, slopeFactors):
    # 4 refers to sorting a 4-variable tuple
    import numpy as np
    # sort in descending order by scores, then by newlabels (within tied scores)
    # (we must do this ascending, and then reverse it)
    # the following assumes labels are int
    dtype2     = [('scores', float), ('newlabels', int), ('labels', int), ('slopeFactors', float)]
    rocTuples2 = list(zip(scores, newlabels, labels, slopeFactors))
    rocArray2  = np.array(rocTuples2, dtype=dtype2)
    temp2      = np.sort(rocArray2, order=['scores', 'newlabels'])
    final2     = temp2[::-1]  # reverse the order of elements
    # put the sorted data back into the original list variables
    scores, newlabels, labels, slopeFactors = zip(*final2)  # zip returns (immutable) tuples
    # we could convert these to lists, but we don't need to: scores = list(scores)
    return scores, newlabels, labels, slopeFactors
#enddef

def showDiscretePartialAUCmeasures(measure_dict, showAllMeasures):
    print(f"{'bAvgA':16s} = {measure_dict['bAvgA']:0.4f}")
    print(f"{'avgSens':16s} = {measure_dict['avgSens']:0.4f}")
    print(f"{'avgSpec':16s} = {measure_dict['avgSpec']:0.4f}")
    print(' ')
    print(f"{'avgPPV':16s} = {measure_dict['avgPPV']:0.4f}")
    print(f"{'avgNPV':16s} = {measure_dict['avgNPV']:0.4f}")
    print(f"{'avgLRp':16s} = {measure_dict['avgLRp']:0.4f}")
    print(f"{'avgLRn':16s} = {measure_dict['avgLRn']:0.4f}")
    print(' ')

    if showAllMeasures:
        print(f"{'avgA':16s} = {measure_dict['avgA']:0.4f}")
        print(f"{'avgBA':16s} = {measure_dict['avgBA']:0.4f}")
        print(' ')
    # endif
#enddef

def showROCinfo(xgroup, ygroup, tgroup, rocRuleLeft, rocRuleRight):
    import numpy as np

    print(f"{'FPR':16s} = [{xgroup[0]:0.3f} {xgroup[1]:0.3f}]")
    print(f"{'TPR':16s} = [{ygroup[0]:0.3f} {ygroup[1]:0.3f}]")

    if np.isinf(tgroup[0]):
        print(f"{'Thresholds':16s} = [{'inf':<5s} {tgroup[1]:0.3f}]")
    else:
        print(f"{'Thresholds':16s} = [{tgroup[0]:0.3f} {tgroup[1]:0.3f}]")
    # endif

    print(f"{' ':33s} {'fpr':<6s} {'tpr':<6s} {'thresh':<6s}      {'fpr':<6s} {'tpr':<6s} {'thresh':<6s}")
    if np.isinf(tgroup[0]):
        print(f"{'rocLeftRight':<16s} = [{rocRuleLeft:<4s} {rocRuleRight:<4s}]   "
              f"({xgroup[0]:0.4f},{ygroup[0]:0.4f},{'inf':<6s}) to "
              f"({xgroup[1]:0.4f},{ygroup[1]:0.4f},{tgroup[1]:0.4f})")
    else:
        print(f"{'rocLeftRight':<16s} = [{rocRuleLeft:<4s} {rocRuleRight:<4s}]   "
              f"({xgroup[0]:0.4f},{ygroup[0]:0.4f},{tgroup[0]:0.4f}) to "
              f"({xgroup[1]:0.4f},{ygroup[1]:0.4f},{tgroup[1]:0.4f})")
    # endif
    print(' ')
    return
#enddef

def showConcordanceMatrixInfo(posIndexC, negIndexC, posScores, negScores,
                              posWeights, negWeights, rocRuleLeft, rocRuleRight):
    p1 = int(posIndexC[0])
    p2 = int(posIndexC[1])
    n1 = int(negIndexC[0])
    n2 = int(negIndexC[1])
    print(f"{'positive indices':16s} = [{p1:2d} {p2:2d}]")
    print(f"{'negative indices':16s} = [{n1:2d} {n2:2d}]")

    print(f"{' ':33s} {'nScore':<6s}  {'ix':<4s},  {'pScore':<6s}  {'ix':<4s}       "
          f"{'nScore':<6s}  {'ix':<4s},  {'pScore':<6s}  {'ix':<4s}")
    print(f"{'rocLeftRight':<16s} = [{rocRuleLeft:<4s} {rocRuleRight:<4s}]   "
          f"({negScores[n1]:0.4f} ({n1:4d}), {posScores[p1]:0.4f} ({p1:4d})) to "
          f"({negScores[n2]:0.4f} ({n2:4d}), {posScores[p2]:0.4f} ({p2:4d}))")
    print(f"with weights: {' ':20s}"
          f"{float(negWeights[n1]):0.4f}{' ':9s}{float(posWeights[p1]):0.4f}{' ':13s}"
          f"{float(negWeights[n2]):0.4f}{' ':9s}{float(posWeights[p2]):0.4f}")
    print(' ')
    return
#enddef

def checkIfGroupsArePerfectCoveringSet(groups, groupAxis):
    import numpy as np

    if groups is not None:
        if groupAxis == 'FPR' or groupAxis == 'TPR' or groupAxis == 'Thresholds' or \
            groupAxis == 'PercentileThresholds':
            sortedGroups = np.sort(np.array(groups), axis=0)
            numGroups    = len(sortedGroups)
            # if the ends are not met
            if sortedGroups[0][0] != 0 or sortedGroups[numGroups-1][1] != 1:
                return False
            # if between groups do not meet perfectly
            for i in range(0, numGroups-1):
                if sortedGroups[i][1] != sortedGroups[i+1][0]:
                    return False
            #endfor
        elif groupAxis == 'Instances':
            # this part needs to be completed, return False by default
            return False
        #endif
        return True
    else:
        return False
    #endif
#enddef

def printGroups(groups):
    numgroups = len(groups)
    print('[ ', end='')
    for i in range(0, numgroups):
        print(f'[{groups[i][0]:0.2f}, {groups[i][1]:0.2f}]', end='')
        if i < numgroups-1:
            print(', ', end='')
    print(' ]')
#enddef