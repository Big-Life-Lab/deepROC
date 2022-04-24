#!/usr/bin/env python
# -*- coding: latin-1 -*-
# FullROC.py
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

from SimpleROC import SimpleROC

class FullROC(SimpleROC):

    def __init__(self, predicted_scores=None, labels=None, poslabel=None, quiet=False):
        '''FullROC constructor. If predicted_scores and labels are empty then it returns an empty object.'''

        #   SimpleROC...        
        super().__init__(predicted_scores=predicted_scores, labels=labels, poslabel=poslabel, quiet=quiet)

        if predicted_scores is not None and labels is not None:
            self.full_fpr, self.full_tpr, self.full_thresholds, self.full_newlabels, self.full_slope_factor \
                = self.construct(self.newlabels, self.predicted_scores, self.newposlabel)
        else:
            self.full_fpr          = None
            self.full_tpr          = None
            self.full_thresholds   = None
            self.full_newlabels    = None
            self.full_slope_factor = None
        #endif
    #enddef

    def construct(self, labels, scores, posclass):
        # Construct the "full" set of points in an ROC curve that correspond to each
        # instance in the data.  This differs from the "standard" ROC points and
        # procedure as outlined in Fawcett, because that procedure skips points
        # which are redundant for a "standard" ROC plot.
        #
        # What are skipped points? In the standard empirical ROC procedure, which
        # creates a staircase plot, if an ROC curve "stair" ascends 3 times,
        # or moves horizontally 4 times, or moves along a slope (tied scores) 3 times,
        # then these intermediary points are not included in the "standard" ROC
        # procedure, with one small exception: points adjacent to (0,0) and (1,1).
        #
        # The "full" ROC procedure is necessary in 2 scenarios:
        #   1) to compute the partial c statistic, and/or
        #   2) to show the full set of actual threshold values along the ROC
        #      curve (in a plot)--which is informative for decision-making.
        #
        # It is necessary because the thresholds at intermediary points do not
        # in general, change in a linear manner, as liner interpolation and/or
        # would erroneously infer. The skipped points from ties in score, are
        # the trouble or complexity. 3 positives and 5 negatives that are all
        # tied in score, are evenly spaced along the diagonal and all have
        # the same score. The left end of a diagonal has a different threshold
        # unless it is at (0,0). When one end has a different threshold,
        # interpolation (which does not apply) would give incorrect results.
        #
        # Notably, the concordant partial AUC (or other partial area measures)
        # work the same with either "full" or "standard" ROC procedures.
        #
        # In variable names, the letter "f" represents "full": e.g., ffpr, ftpr.
        import numpy as np
        import math
        from Helpers.ROCFunctions import sortScoresFixLabels
        scores, newlabels, labels = sortScoresFixLabels(scores, labels, posclass, False)  # False = descending

        n             = len(labels) + 1  # +1 for the (0,0) point
        finalIndex    = n - 1
        blank         = np.zeros(n)
        ffpr          = blank.copy()
        ftpr          = blank.copy()
        fthresh       = blank.copy()
        fSlopeFactor  = np.ones(n).copy()
        fnewlabel     = np.array(newlabels)

        # for score, newlabel, label in zip(scores, newlabels, labels)
        thisFP        = 0
        thisTP        = 0
        numTP         = len(np.where(np.array(newlabels) == 1)[0])
        numFP         = len(np.where(np.array(newlabels) == 0)[0])
        tPrev         = math.inf  # previous threshold
        numTies       = 0
        numPosTies    = 0
        firstTieIndex = 0

        def addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, tPrev):
            # print(f'numFP, numTP, len(ffpr), index, tPrev = {numFP, numTP, len(ffpr), index, tPrev}')
            ftpr[index]    = thisTP / numTP  # first time here is (0, 0) with thresh Inf
            ffpr[index]    = thisFP / numFP
            fthresh[index] = tPrev
            return ffpr, ftpr, fthresh

        # enddef

        def addROCtie(ffpr, ftpr, fthresh, fSlopeFactor, index, tPrev, numTies, numPosTies, firstTieIndex):
            rise = ftpr[index] - ftpr[firstTieIndex - 1]  # incl. point before
            run  = ffpr[index] - ffpr[firstTieIndex - 1]  # incl. point before
            if rise > 0 and run > 0:  # if ties cause a sloped ROC line segment
                # then capture the slope of the line segment, by its rise and run,
                # for each instance along the segment
                thisRise = numPosTies / numTies
                # thisRun = 1 - thisRise
                # i.e., whereas non-sloped segments travel 1 instance unit along only 1 of the axes
                # a sloped segment travels thisRise (or fRise) units along the y-axis and thisRun
                # units along the x-axis.
            else:
                thisRise = 1
            # endif
            for i in np.arange(0, numTies):
                ftpr[firstTieIndex + i]    = ftpr[firstTieIndex - 1] + rise * ((i + 1) / numTies)
                ffpr[firstTieIndex + i]    = ffpr[firstTieIndex - 1] + run  * ((i + 1) / numTies)
                fthresh[firstTieIndex + i] = tPrev
                # Notably, the slope factor is only stored for points along a sloped line segment
                # otherwise it defaults to a factor of 1 (no effect).
                # Also, the slope factor indicates the slope of the line before the current point
                # so the first point takes the default value
                if i > 0:
                    fSlopeFactor[firstTieIndex + i] = thisRise
                # endif
            # endfor
            return ffpr, ftpr, fthresh, fSlopeFactor

        # enddef

        index = 0  # index of all/full ROC points including hidden ties
        # add (0,0) point
        ffpr, ftpr, fthresh = addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, tPrev)
        fnewlabel = np.insert(fnewlabel, 0, -1)
        index = 1

        for score, newlabel in zip(scores, newlabels):
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
                thisTP = thisTP + 1
            else:
                thisFP = thisFP + 1
            # endif
            #
            if index == finalIndex:  # (a) last roc point
                tied_w_next = False
            else:
                tied_w_next = (score == scores[index])  # not "index + 1" because scores has no (0,0) point
            # endif
            tied_w_prev = (score == tPrev)
            if not tied_w_prev and not tied_w_next:  # (b) not a tie
                ffpr, ftpr, fthresh = addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, score)
                tPrev      = score  # set previous threshold for next iteration to current score
                numTies    = 0
                numPosTies = 0
            elif not tied_w_prev and tied_w_next:  # (c) new tie
                ffpr, ftpr, fthresh = addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, score)
                firstTieIndex = index
                numTies = 1
                if newlabel == 1:
                    numPosTies = 1
                tPrev = score  # set previous threshold for next iteration to current score
            elif tied_w_prev and tied_w_next:  # (d) middle tie
                numTies = numTies + 1
                if newlabel == 1:
                    numPosTies = numPosTies + 1
            elif tied_w_prev and not tied_w_next:  # (e) last tie (in current series)
                ffpr, ftpr, fthresh = addStandardROCpoint(ffpr, ftpr, fthresh, thisTP, thisFP, index, score)
                numTies = numTies + 1
                if newlabel == 1:
                    numPosTies = numPosTies + 1
                ffpr, ftpr, fthresh, fSlopeFactor = addROCtie(ffpr, ftpr, fthresh, fSlopeFactor, index, score,
                                                              numTies, numPosTies, firstTieIndex)
            # endif
            index = index + 1  # increment index for next iteration
        # endfor

        ffpr         = ffpr[:index + 1].copy()
        ftpr         = ftpr[:index + 1].copy()
        fthresh      = fthresh[:index + 1].copy()
        fSlopeFactor = fSlopeFactor[:index + 1].copy()

        return ffpr, ftpr, fthresh, fnewlabel, fSlopeFactor
    # enddef

    def get(self):
        '''get returns the full ROC arrays for predicted_scores, labels, full fpr,
           full tpr, full thresholds.'''

        # if self.predicted_scores is not None and self.newlabels is not None:
        #     if self.full_fpr is None and full_tpr is None:
        #         self.full_fpr, self.full_tpr, self.full_thresholds, \
        #         self.slopeFactor, slope.max_threshold = tbd(self.newlabels,
        #                                                     self.predicted_scores,
        #                                                     self.poslabel)
        #     #endif
        # else:
        # #endif
        # note: we return newlabels in the following
        return self.predicted_scores, self.newlabels, self.full_fpr, self.full_tpr, \
               self.full_thresholds, self.full_slope_factor
    #enddef

    def getAUC(self):
        ''' Computes and returns the AUC or AUROC (a continuous measure)'''
        from sklearn import metrics

        if self.predicted_scores is None and self.newlabels is None:
            if self.full_fpr is None and self.full_tpr is None:
                SystemError('Predicted scores and labels, or full FPR and full TPR, are required to ' +
                            'compute the AUC for full ROC data.')
            else:
                self.AUC = metrics.auc(self.full_fpr, self.full_tpr)
            #endif
        else:  # self.predicted_scores and self.newlabels are populated
            self.AUC = metrics.roc_auc_score(self.newlabels, self.predicted_scores)
        #endif
        return self.AUC
    #enddef

    # uses getC() from the superclass SimpleROC

    def set_fpr_tpr(self, fpr=None, tpr=None):
        '''The set_fpr_tpr method is not allowed for FullROC.'''
        SystemError('set_fpr_tpr is not allowed for FullROC, because FullROC depends on the integrity ' + \
                    'of properly spaced points derived from scores and labels.')
    #enddef

    def plot(self, plotTitle, showThresholds=True, showOptimalROCpoints=True, costs=None,
             saveFileName=None, numShowThresh=30, showPlot=True, labelThresh=True, full_fpr_tpr=True):
             return super().plot(plotTitle, showThresholds, showOptimalROCpoints, costs,
                                 saveFileName, numShowThresh, showPlot, labelThresh, full_fpr_tpr)
    #enddef

    # def plot(self, plotTitle, showThresholds=True, showOptimalROCpoints=True, saveFileName=None,
    #          numShowThresholds=30, showPlot=True):
    #     '''plot provides an ROC plot with full data (including a point for each tie), and
    #        optional labels for threshold percentiles or thresholds, and optional optimal ROC points
    #        based on Metz (or a fully generalized Youden's index.'''
    #     from ROCPlot import plotROC
    #     from ROCPlot import plotOpt
    #     import matplotlib.pyplot as plt
    #
    #     fancyLabel = True
    #     fig, ax = plotROC(self.full_fpr, self.full_tpr, plotTitle, numShowThresholds,
    #                       self.full_thresholds, fancyLabel)
    #     if showOptimalROCpoints:
    #         # get optimal points here...
    #         plotOpt(fpr_opt, tpr_opt, thresh_opt, maxThresh, fancyLabel)  # add the optimal ROC points
    #     if showPlot:
    #         plt.show()
    #     #modeShort = mode[:-3]  # training -> train, testing -> test
    #     #fig.savefig(f'output/ROC_{modeShort}_{testNum}-{index}.png')
    #     if saveFileName is not None:
    #         fig.savefig(saveFileName)
    #
    #     return False
    # #enddef

    # def plotConcordanceMatrix(self):
    #     '''plots a Concordance Matrix with full data (including a point for each tie).'''
    #     from ConcordanceMatrix import ConcordanceMatrix
    #     aConcordanceMatrix = ConcordanceMatrix(predicted_scores=self.predicted_scores, labels=self.labels)
    #     aConcordanceMatrix.plot
    #     # to be completed...
    #     return False
    # #enddef
        
    def __str__(self):
        '''This method prints the object as a string of its content re 
           predicted scores, labels, full fpr, full tpr, full thresholds.'''
        
        rocdata = f'score, label, fullfpr, fulltpr, fullthr\n'
        for a, b, c, d, e in zip(self.predicted_scores, self.labels,
                                 self.full_fpr, self.full_tpr, self.full_thresholds):
            rocdata = rocdata + f'{a:0.3f}, {b:<5d}, {c:0.3f}  , {d:0.3f}  , {e:0.3f}\n'
        #endfor
        #print(rocdata)
        return rocdata
    #enddef