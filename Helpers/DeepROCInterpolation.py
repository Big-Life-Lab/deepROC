#!/usr/bin/env python
# -*- coding: latin-1 -*-
# DeepROCInterpolation.py
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

# get_Match_or_Interpolation_Points(rstat, endpoint)
# interpolateROC(rstat, ostat, thresh, ixL, ixR, interpValue)
# getClosestGroups(groups, groupAxis, ffpr, ftpr, fthresh)
# partial_C_statistic_interpolated(posScores, negScores, posWeights, negWeights, ygroups, xgroups):

# imports are locally defined in each function

def get_Match_or_Interpolation_Points(rstat, endpoint):
    ''' returns ix, ixL, ixR '''
    import numpy as np

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

def getClosestGroups(groups, groupAxis, ffpr, ftpr, fthresh):
    if groupAxis == 'FPR':
        axisValues  = ffpr
        otherValues = ftpr
    elif groupAxis == 'TPR':
        axisValues  = ftpr
        otherValues = ffpr
    else:
        raise NotImplementedError
    # endif

    for i in [0, 1]:
        ix, ixL, ixR = get_Match_or_Interpolation_Points(axisValues, groups[i])
        if len(ix) == 0:  # if no exact match, then use closest point
            _, __, perCentFromLeft = interpolateROC(axisValues, otherValues, fthresh, ixL, ixR, groups[i])
            if perCentFromLeft > 0.5:
                groups[i] = axisValues[ixR]
            else:
                groups[i] = axisValues[ixL]
            # endif
        # endif
    # endfor
    return groups
# enddef

def partial_C_statistic_interpolated(posScores, negScores, posWeights, negWeights, ygroups, xgroups):
    ''' partial_c_statistic computes the cStatistic given a vector of scores for actual
        positives and a vector of scores for actual negatives in the ROC data and
        corresponding weight vectors with elements in [0,1] that indicate which instances
        are in the partial range and how much weight for each instance.  That is,
        boundary instances may have partial weight, interior instances will have weight 1
        and exterior instances will have weight 0'''

    # NOTE: THERE IS ONE CASE NOT WRITTEN/HANDLED YET (test dataset 12) WHEN THE TOP AND
    # BOTTOM (OR LEFT AND RIGHT) BOUNDS OF THE PARTIAL CURVE ARE WITHIN THE SAME CELL,
    # THE LOGIC DOES NOT YET HANDLE THIS PROPERLY.  TIME IS THE ONLY LIMITATION...
    from Helpers.DeepROCFunctions import Heaviside

    P      = len(posScores)
    N      = len(negScores)
    partP  = float(sum(posWeights))
    partN  = float(sum(negWeights))

    # for the case of a flat vertical range, or a flat horizontal range, the scores and
    # weights do not tell us enough to compute the partial c statistic, so we need
    # xgroups and ygroups which are indices including decimals for interpolation
    if partP == 0:  # if range in horizontal, no vertical (no positives)
        posIndexC_0_1   = int(ygroups[0])      # round down to int, for index
        remaining_posWeight = float(ygroups[0]) - posIndexC_0_1
    #endif
    if partN == 0:  # if range in vertical, no horizontal (no negatives)
        negIndexC_0_1   = int(N-float(xgroups[0]))  # round down to int, for index
        remaining_negWeight = N-float(xgroups[0]) - negIndexC_0_1
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

    # compute positive count count_x (in the horizontal stripe) based on:
    #   the subset of positives in the partial curve (non-zero posWeights)
    #   the negatives to the right of the partial curve (negWeights with indices
    #     higher than or equal to the horizontally leftmost non-zero negWeights index)
    if topIndex == -1 or bottomIndex == -1:  # no horizontal stripe
        count_x = 0
    elif partN == 0:  # horizontal stripe but no vertical stripe and no negatives
        count_x = 0
        for k in range(negIndexC_0_1, N):  # enforce under curve positives (no partial index)
            for j in range(0, P):
                if posWeights[j] == 0:     # enforce subset of positives
                    continue
                #endif
                # outside subset of negatives use one weight, no h
                count_x = count_x + float(posWeights[j])
            # endfor
        # endfor
        count_x = count_x + (remaining_negWeight * partP)  # add partial negative index/weight
    else:
        count_x = 0
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
                    count_x = count_x + posW * negW * h
                else:  # outside subset of negatives use one weight, no h
                    count_x = count_x + posW
                if k == rightIndex and 0 < negW < 1:  # in this case
                    count_x = count_x + posW * (1 - negW)      # also add this part
                #endif
            #endfor
        #endfor
    #endif

    # compute negative count count_y (in the vertical stripe) based on:
    #   the subset of negatives in the partial curve (non-zero negWeights)
    #   the positives under the partial curve (posWeights with indices
    #     lower than or equal to the vertically topmost non-zero posWeights index)
    #     note: posWeights is directional, so if at the bottom of a range,
    #           it must be reversed: 1-posWeights
    if leftIndex == -1 or rightIndex == -1:  # no vertical stripe
        count_y = 0
    elif partP == 0:  # vertical stripe but no horizontal stripe and no positives
        count_y = 0
        for j in range(0, posIndexC_0_1+1):  # enforce under curve positives, no partial index
            for k in range(0, N):
                if negWeights[k] == 0:           # enforce subset of negatives
                    continue
                # outside subset of positives use one weight, no h
                count_y = count_y + float(negWeights[k])
            # endfor
        # endfor
        count_y = count_y + (remaining_posWeight * partN)  # add partial positive index/weight
    else:  # vertical stripe normal case
        count_y = 0
        for j in range(0, topIndex+1):  # enforce under curve positives
            for k in range(0, N):
                if negWeights[k] == 0:  # enforce subset of negatives
                    continue
                # h==1 correct rank, h==0.5 tie, h==0 wrong rank
                h    = Heaviside(float(posScores[j]) - float(negScores[k]))
                posW = float(posWeights[j])
                negW = float(negWeights[k])
                if posW > 0:  # within subset of positives use both weights and h
                    count_y = count_y + negW * posW * h
                else:  # outside subset of positives use one weight, no h
                    count_y = count_y + negW
                if j == bottomIndex and 0 < posW < 1:  # in this case
                    count_y = count_y + negW * (1 - posW)        # also add this part
                #endif
            #endfor
        #endfor
    #endif

    cLocal_i = 0
    for j in range(0, P):
        for k in range(0, N):
            h      = Heaviside(float(posScores[j]) - float(negScores[k]))
            cLocal_i = cLocal_i + float(posWeights[j]) * \
                       float(negWeights[k]) * h
        #endfor
    #endfor

    whole_area = N * P
    if whole_area == 0:
        C_i       = 0
        C_area_y  = 0
        C_area_x  = 0
    else:
        C_i       = (1/2)*(count_x/whole_area) + (1/2)*(count_y/whole_area)
        C_area_y  = count_y / whole_area
        C_area_x  = count_x / whole_area
    #endif

    horizontal_stripe_area = N * partP
    vertical_stripe_area   = P * partN

    if   horizontal_stripe_area == 0 and vertical_stripe_area == 0:
        Cn_i = 0
    else:
        Cn_i = (C_i * (2*whole_area)) / (horizontal_stripe_area + vertical_stripe_area)
        #Cn_i = (C_i * 2) / (horizontal_stripe_area + vertical_stripe_area)
    #endif

    if horizontal_stripe_area == 0:
        C_avgSpec = 0
    else:
        C_avgSpec = count_x/horizontal_stripe_area
    #endif

    if vertical_stripe_area == 0:
        C_avgSens = 0
    else:
        C_avgSens = count_y/vertical_stripe_area
    #endif

    local_area = partP * partN
    if local_area == 0:
        cLocal_i = 0
    else:
        cLocal_i = cLocal_i / local_area
    #endif

    #return C_i, Cn_i, cLocal_i, count_x/count_area, count_y/count_area, C_avgSpec, C_avgSens, count_x, count_y, \
    #       count_area, horizontal_stripe_area, vertical_stripe_area

    #       C_i, Cn_i, cLocal_i, count_x1, count_y1, count_x2, count_y2, count_x, count_y, \
    #       whole_area, horizontal_stripe_area, vertical_stripe_area \
    #            = partial_c_statistic(posScores, negScores, posWeights, negWeights, posIndexC, negIndexC)
    group_measures_dict = {'C_i':         C_i,            # partial C
                           'Cn_i':        Cn_i,           # partial C normalized
                           'CLocal_i':    cLocal_i,       # cLocal
                           'C_area_y':    C_area_y,       # area:     cn1 = cp/whole_area
                           'C_area_x':    C_area_x,       # area:     cp1 = cn/whole_area
                           'C_avgSens':   C_avgSens,      # distance: cn2 = cn_normalized
                           'C_avgSpec':   C_avgSpec,      # distance: cp2 = cp_normalized
                           'Uy':          count_y,        # count:    cn  (U statistic y part)
                           'Ux':          count_x,        # count:    cp  (U statistic x part)
                           'whole_area':  whole_area,     # count:    whole_area
                           'horizontal_stripe_area': horizontal_stripe_area,    
                                                          # count:    horizontal_stripe_area
                           'vertical_stripe_area':   vertical_stripe_area     
                                                          # count:    vertical_stripe_area
                          }
    return group_measures_dict
#enddef
