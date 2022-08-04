#!/usr/bin/env python
# -*- coding: latin-1 -*-
# ConcordanceMatrixPlot.py
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

class ConcordanceMatrixPlot(object):

    def __init__(self, ROCdata):
        '''ConcordanceMatrix constructor that takes as input a SimpleROC, FullROC or DeepROC object.'''
        import numpy as np
        from Helpers.ROCFunctions     import sortScoresAndLabels2
        from Helpers.DeepROCFunctions import sortScoresAndLabels3

        if ROCdata is not None:
            self.ROCdata = ROCdata
            theData      = self.ROCdata.get()
            if   len(theData) == 5:
                self.predicted_scores, self.newlabels, self.fpr, self.tpr, self.thresholds = theData
                if self.fpr is None:
                    SystemError('The ROCdata object cannot be empty.')
                self.poslabel         = 1
                self.slope_factor     = None
                self.predicted_scores, self.newlabels = sortScoresAndLabels2(self.predicted_scores, self.newlabels)

                # np.argwhere only works on np.array
                posIndices = np.argwhere(np.array(self.newlabels) == 1).flatten()
                negIndices = np.argwhere(np.array(self.newlabels) == 0).flatten()
                self.posScores        = list(np.array(self.predicted_scores)[posIndices])
                self.negScores        = list(np.array(self.predicted_scores)[negIndices])

            elif len(theData) == 6:
                self.predicted_scores, self.newlabels, self.fpr, self.tpr, self.thresholds, self.slope_factor = theData
                if self.fpr is None:
                    SystemError('The ROCdata object cannot be empty.')
                self.poslabel         = 1
                self.predicted_scores, self.newlabels, self.slope_factor = sortScoresAndLabels3(self.predicted_scores,
                                                                                                self.newlabels,
                                                                                                self.slope_factor)
                # np.argwhere only works on np.array
                posIndices = np.argwhere(np.array(self.newlabels) == 1).flatten()
                negIndices = np.argwhere(np.array(self.newlabels) == 0).flatten()
                self.posScores        = list(np.array(self.predicted_scores)[posIndices])
                self.negScores        = list(np.array(self.predicted_scores)[negIndices])
            else:
                SystemError('Unexpected return values from ROCdata.get().')
            #endif
        else:
            SystemError('The ROCdata object cannot be empty.')
        #endif
    #enddef

    def setGroupsBy(self, groupAxis=None, groups=None, groupByClosestInstance=False):
        return self.ROCdata.setGroupsBy(groupAxis, groups, groupByClosestInstance)
    #enddef


    def plotGroup(self, plotTitle=None, groupIndex=None, showError=False, showThresholds=True, showOptimalROCpoints=True, costs=None,
                 saveFileName=None, numShowThresholds=30, showPlot=True, labelThresh=True):
        from Helpers.DeepROCPlot import plotPartialArea

        print('This function has a bug, to be fixed.')

        if self.ROCdata.__class__.__name__ != 'DeepROC':
            SystemError('To plot a group please define the ConcordanceMatrixPlot with a DeepROC object.')

        if self.ROCdata.groups is None:
            SystemError('Cannot plotGroup until groups are set with setGroupsBy.')

        # get group information for AUC
        partial_full_fpr, partial_full_tpr, rangeEndpoints1, groupByOtherAxis, groupByThreshold, \
          matchedIndices, approxIndices, group, rocRuleLeft, rocRuleRight = \
          self.ROCdata.getGroupForAUCi(groupIndex, forFolds=False)

        self.ROCdata.plotGroup(plotTitle, groupIndex, showError, showThresholds,
                               showOptimalROCpoints, costs, saveFileName, numShowThresholds,
                               showPlot, labelThresh)

        # add fills for partial areas (clobbers points, score labels)
        plotPartialArea(partial_full_fpr, partial_full_tpr, showError)

        return
    #enddef

    def plot(self, plotTitle, showThresholds=True, showOptimalROCpoints=True, costs=None,
             saveFileName=None, numShowThresholds=30, showPlot=True, labelThresh=True):
        '''Plots the Concordance Matrix.'''
        import matplotlib.pyplot    as     plt
        from   Helpers.ROCPlot      import addPointsAndLabels
        from   Helpers.ROCPlot      import plotOptimalPointWithThreshold
        from   Helpers.ROCFunctions import getSlopeOrSkew
        from   Helpers.ROCFunctions import optimal_ROC_point_indices
        import math

        if self.fpr is None:
            SystemError('Cannot plot empty ConcordanceMatrix object.')

        def main_plot_logic(plotTitle, maxInstancesPerAxis):
            ''' create the plot and return the figure, the axis, the number of negatives shown
                along the x axis and the number of positives shown along the y axis.'''
            import math
            import matplotlib.ticker as ticker

            # ordinal function from:
            # https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement  which adapts Gareth's answer from:
            # https://codegolf.stackexchange.com/questions/4707/outputting-ordinal-numbers-1st-2nd-3rd#answer-4712
            ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

            negScores = list(self.negScores)
            posScores = list(self.posScores)
            ntotal = len(negScores)
            ptotal = len(posScores)
            NPclassRatio = ntotal / ptotal
            nfactor = 1
            pfactor = 1
            nsuffix = ''
            psuffix = ''

            if ntotal > maxInstancesPerAxis:
                nfactor = ntotal / maxInstancesPerAxis
                nfactortxt = int(round(nfactor))
                nsuffix = f' (approx. every {ordinal(nfactortxt)})'
            # endif
            if ptotal > maxInstancesPerAxis:
                pfactor = ptotal / maxInstancesPerAxis
                pfactortxt = int(round(pfactor))
                psuffix = f' (approx. every {ordinal(pfactortxt)})'
            # endif
            if nfactor > 1 and pfactor > 1:
                maxfactor = max(nfactor, pfactor)
                maxfactortxt = int(round(maxfactor))
                nfactor = maxfactor
                pfactor = maxfactor
                nsuffix = f' (approx. every {ordinal(maxfactortxt)})'
                psuffix = f' (approx. every {ordinal(maxfactortxt)})'
            # endif

            show_ntotal = int(math.floor(ntotal / nfactor))  # in the plot shown we round-off (ignore) a bit at the end
            show_ptotal = int(math.floor(ptotal / pfactor))  # in the plot shown we round-off (ignore) a bit at the end

            # create plot with ROC curve
            fig = plt.figure()
            ax  = fig.add_subplot(1, 1, 1, xticklabels=[], yticklabels=[])
            plt.plot(self.fpr, self.tpr, color='blue', lw=2)

            # label negative instances
            for i in range(0, show_ntotal):
                idx = int(round(i * nfactor))
                x = (i + 0.5) / show_ntotal  # half indicates the center of the column in the concordance matrix
                score = float(negScores[idx])
                label, sizexy, fontsize = self.get_cMatrix_Label_Size_Fontsize(score)
                offset = (x - (0.5 * sizexy[0]), -2 - sizexy[1])
                plt.annotate(label, (x, 0), textcoords="offset points",
                             xytext=offset, ha='left', fontsize=fontsize)
            # endfor

            # label positive instances
            for i in range(0, show_ptotal):
                idx = int(round(i * pfactor))
                y = (i + 0.5) / show_ptotal  # half indicates the center of the column in the concordance matrix
                score = float(posScores[idx])
                label, sizexy, fontsize = self.get_cMatrix_Label_Size_Fontsize(score)
                offset = (-2 - sizexy[0], y - (0.5 * sizexy[1]))
                plt.annotate(label, (0, y), textcoords="offset points",
                             xytext=offset, ha='left', fontsize=fontsize)
            # endfor

            plt.xlabel(f' \nNegative Instances{nsuffix}')
            plt.ylabel(f'Positive Instances{psuffix}\n \n ')
            plt.title(plotTitle)
            plt.xlim(-0.01, 1.0)
            plt.ylim(0.0, 1.01)
            ax.axes.xaxis.set_major_locator(ticker.MultipleLocator(1 / show_ntotal))
            ax.axes.yaxis.set_major_locator(ticker.MultipleLocator(1 / show_ptotal))
            plt.grid(True)
            return fig, ax, show_ntotal, show_ptotal, NPclassRatio
        # enddef

        maxInstancesPerAxis = 20
        fig, ax, show_ntotal, show_ptotal, NPclassRatio = main_plot_logic(plotTitle, maxInstancesPerAxis)

        # add points, score labels, optimal ROC points
        fancyLabel   = True
        if showThresholds:
            addPointsAndLabels(self.fpr, self.tpr, numShowThresholds, self.thresholds, labelThresh)
        #endif

        if showOptimalROCpoints:
            skew         = getSlopeOrSkew(NPclassRatio=NPclassRatio, costs=costs, quiet=True)
            opt_indices  = optimal_ROC_point_indices(self.fpr, self.tpr, skew)
            # for plotOpt...
            if not math.isinf(self.thresholds[0]):
                maxThreshold = self.thresholds[0]  # if first (max) thresh is not infinite, then use it for label
            else:
                maxThreshold = self.thresholds[1]  # otherwise, use the next label which should be finite
            # endif
            plotOptimalPointWithThreshold(self.fpr[opt_indices], self.tpr[opt_indices], self.thresholds[opt_indices],
                                          maxThreshold, fancyLabel)
        #endif

        if showPlot:
            plt.show()

        if saveFileName is not None:
            fig.savefig(saveFileName)

        return fig, ax
    #enddef

    def get_cMatrix_Label_Size_Fontsize(self, val):
        ''' returns label, sizexy, fontsize '''
        import numpy as np

        # label offsets are dependent on the fontsize
        fontsize = 'x-small'
        # fontsize: float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        # number formatting
        if val < 10:
            # label   = "{:.2f}".format(val)
            label = f'{val:.2g}'
        elif np.isinf(val):
            if val > 0:
                label = 'inf'
            else:
                label = '-inf'
            # endif
        else:
            val = int(round(val))
            label = f'{val:.2d}'
        # endif
        if val < 1 and val > 0:  # if decimal included
            numberWidth = 5 * (len(label) - 1)
            decimalWidth = 1
            width = numberWidth + decimalWidth
        else:
            width = 5 * len(label)
        # endif
        sizexy = (width, 5)
        return label, sizexy, fontsize
    # enddef