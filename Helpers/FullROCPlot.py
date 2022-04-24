#!/usr/bin/env python
# -*- coding: latin-1 -*-
# FullROCPlot.py
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

# plotConcordanceMatrix(fpr, tpr, negScores, posScores, plotTitle, maxInstancesPerAxis)
# get_cMatrix_Label_Size_Fontsize(val)

# imports are locally defined in each function
def plotConcordanceMatrix(fpr, tpr, negScores, posScores, plotTitle, maxInstancesPerAxis):
    ''' returns fig, ax '''
    import math
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

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
    plt.xlim(-0.01, 1.0)
    plt.ylim(0.0, 1.01)
    ax.axes.xaxis.set_major_locator(ticker.MultipleLocator(1/show_ntotal))
    ax.axes.yaxis.set_major_locator(ticker.MultipleLocator(1/show_ptotal))
    plt.grid(True)
    return fig, ax, show_ntotal, show_ptotal
#enddef

def get_cMatrix_Label_Size_Fontsize(val):
    ''' returns label, sizexy, fontsize '''
    import numpy as np

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

def addPoints(fpr, tpr, numThresh, thresh, fancyLabel):
    # add threshold labels and circles
    # allow up to numThresh labels per plot, or numThresh+4 in the first multiple
    import math
    import matplotlib.pyplot as plt

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


def get_ROC_Curve_Label_Offset_Fontsize(x, y, t, maxThresh, fancyLabel):
    import math

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