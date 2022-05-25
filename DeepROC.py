#!/usr/bin/env python
# -*- coding: latin-1 -*-
# DeepROC.py
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

from FullROC           import FullROC

class DeepROC(FullROC):

    def __init__(self, predicted_scores=None, labels=None, poslabel=None, quiet=False):
        '''DeepROC constructor. If predicted_scores and labels are
           empty then it returns an empty object.'''
        
        super().__init__(predicted_scores=predicted_scores, labels=labels, poslabel=poslabel, quiet=quiet)

        #   Deep ROC...
        self.interpolation     = None
        self.populationPrevalence = None

        #   Analysis...
        self.wholeMeasures     = None
        self.groupMeasures     = None
        self.pointMeasures     = None
        
        self.groups            = None
        self.groupAxis         = None
        self.groupsArePerfectCoveringSet = None
        
        self.wholeResults      = None
        self.groupResults      = None
        self.pointResults      = None

        self.foldsNPclassRatio = None
        self.NPclassRatio      = None
        self.priorPoint        = None
    #enddef

    def setPopulationPrevalence(self, populationPrevalence=None):
        self.populationPrevalence = populationPrevalence
    #enddef

    def setGroupsBy(self, groupAxis=None, groups=None, groupByClosestInstance=False):
        '''setGroupsBy interpolates instances by default unless groupAxis is Instances or
           groupByClosestInstance is set to True. It returns the groups used.'''
        from Helpers.DeepROCFunctions import checkIfGroupsArePerfectCoveringSet

        if groupAxis is None or (groupAxis != 'TPR' and
                                 groupAxis != 'FPR' and
                                 groupAxis != 'Thresholds' and
                                 groupAxis != 'PercentileThresholds' and
                                 groupAxis != 'Instances'
                                 ):
            SystemError(f'groupAxis {groupAxis} is not recognized.')
        #endif
         
        self.groupAxis         = groupAxis
        self.group_measures    = None       # clear any previous configuration

        if groupByClosestInstance:
            self.groups        = self.getClosestGroups(groups)
            self.interpolation = False
            
        elif groupAxis == 'Instances':
            self.groups        = groups
            self.interpolation = False
            
        else:
            self.groups        = groups
            self.interpolation = True
        #endif
        
        if checkIfGroupsArePerfectCoveringSet(groups, groupAxis):
            # print(f'The groups cover {groupAxis} [0,1] and are non-overlapping (a perfect covering set).')
            self.groupsArePerfectCoveringSet = True
        else:
            # print(f'The groups are not a perfect covering set over {groupAxis} [0,1]')
            self.groupsArePerfectCoveringSet = False
        #endif

        return self.groups
    #enddef

    def getGroupForAUCi(self, groupIndex, forFolds):
        import numpy as np
        if self.groups is None:
            SystemError('Must use setGroupsBy() method first.')
        #endif

        # get group information for AUC
        if self.groupAxis == 'TPR' or self.groupAxis == 'FPR':
            if ( self.groupAxis == 'FPR' and self.groups[groupIndex][0] == 0) or \
               ( self.groupAxis == 'TPR' and self.groups[groupIndex][1] == 0):
                rocRuleLeft  = 'SW'
                rocRuleRight = 'NE'
            else:
                rocRuleLeft  = 'NE'
                rocRuleRight = 'NE'
            # endif

            quiet2 = True
            group  = self.groups[groupIndex]
            if forFolds:
                fpr        = self.mean_fpr
                tpr        = self.mean_tpr
                thresholds = np.ones(self.mean_fpr.shape)
            else:
                fpr        = self.full_fpr
                tpr        = self.full_tpr
                thresholds = self.full_thresholds
            #endif
            partial_fpr, partial_tpr, groupByOtherAxis, groupByThreshold, \
                matchedIndices, approxIndices \
                = self.getGroupForAUC(fpr, tpr, thresholds, self.groupAxis, group,
                                      rocRuleLeft, rocRuleRight, quiet2)
            return partial_fpr, partial_tpr, groupByOtherAxis, groupByThreshold, \
                   matchedIndices, approxIndices, group, rocRuleLeft, rocRuleRight
        else:
            # to be completed
            return None, None, None, None, None, \
                   None, None, None, None
        #endif
    #enddef

    def analyzeGroup(self, groupIndex, showData=False, forFolds=False, quiet=False):
        from Helpers.DeepROCFunctions import partial_C_statistic_simple
        from Helpers.DeepROCFunctions import partial_C_statistic
        from Helpers.DeepROCFunctions import concordant_partial_AUC
        from Helpers.DeepROCFunctions import discrete_partial_roc_measures
        from Helpers.DeepROCFunctions import partial_area_index_proxy
        from Helpers.DeepROCFunctions import areEpsilonEqual
        from Helpers.DeepROCFunctions import showConcordanceMatrixInfo
        from Helpers.DeepROCFunctions import showCmeasures
        from Helpers.DeepROCFunctions import showAUCmeasures
        from Helpers.DeepROCFunctions import showWholeAUCmeasures
        from Helpers.DeepROCFunctions import showDiscretePartialAUCmeasures
        from Helpers.DeepROCFunctions import showROCinfo
        import numpy as np
        import sklearn.metrics as metrics

        if self.groups is None:
            SystemError('Must use setGroupsBy() method first.')
        #endif

        # get group information for AUC
        partial_fpr, partial_tpr, groupByOtherAxis, groupByThreshold, \
          matchedIndices, approxIndices, rangeEndpoints1, rocRuleLeft, rocRuleRight = \
          self.getGroupForAUCi(groupIndex, forFolds=forFolds)

        # setup variables for printing group information
        if self.groupAxis == 'FPR':
            xgroup = self.groups[groupIndex]
            ygroup = groupByOtherAxis
        else:
            ygroup = self.groups[groupIndex]
            xgroup = groupByOtherAxis
        # endif
        tgroup = groupByThreshold

        # Show the partial ROC data
        quiet2 = True
        if showData and not quiet2:
            print(f"{'pfpr':6s}, {'ptpr':6s}")
            for x, y in zip(partial_fpr, partial_tpr):
                print(f'{x:-6.3g}, {y:-6.3g}')
            # endfor
            print(' ')
        # endif

        # get group information for C
        if forFolds:
            # for Mean ROC we cannot compute C measures

            # show ROC group boundaries and information
            if not quiet:
                showROCinfo(xgroup, ygroup, tgroup, rocRuleLeft, rocRuleRight)
            # endif

            measure_dict = dict()

        else:
            negIndexC, posIndexC, negScores, posScores, negWeights, posWeights, negWidths, posHeights \
                = self.getGroupForC(self.full_fpr, self.full_tpr, self.full_thresholds,
                                    self.full_slope_factor, self.predicted_scores, self.full_newlabels,
                                    self.newposlabel, xgroup, ygroup)
            # show ROC group boundaries and information
            if not quiet:
                showROCinfo(xgroup, ygroup, tgroup, rocRuleLeft, rocRuleRight)
            #endif
            # show Concordance Matrix group boundaries
            if not quiet:
                showConcordanceMatrixInfo(posIndexC, negIndexC, posScores, negScores,
                                          posWeights, negWeights, rocRuleLeft, rocRuleRight)
            #endif

            measure_dict = dict()

            # compute measures related to partial C statistic
            if not self.interpolation:
                temp_dict1 = partial_C_statistic_simple(posScores, negScores, posWeights, negWeights)
                temp_dict2 = dict(C_local=None, Ux_count=None, Uy_count=None)
                # update measure_dict with results
                measure_dict.update(temp_dict1)
                measure_dict.update(temp_dict2)
                # also store results in local variables
                C_i, C_area_y,  C_area_x, Cn_i, Cn_avgSens, Cn_avgSpec, C_local, Ux_count, Uy_count = \
                    [measure_dict[key] for key in ['C_i', 'C_area_y', 'C_area_x', 'Cn_i', 'Cn_avgSens',
                                                   'Cn_avgSpec', 'C_local', 'Ux_count', 'Uy_count']]
                # supporting variables
                whole_area, vertical_stripe_area, horizontal_stripe_area = None, None, None
            else:
                temp_dict, whole_area, horizontal_stripe_area, vertical_stripe_area = \
                    partial_C_statistic(posScores, negScores, posWeights, negWeights, posIndexC, negIndexC)
                # update measure_dict with results
                measure_dict.update(temp_dict)
                # also store results in local variables
                C_i, C_area_y,  C_area_x, Cn_i, Cn_avgSens, Cn_avgSpec, C_local, Ux_count, Uy_count = \
                    [measure_dict[key] for key in ['C_i', 'C_area_y', 'C_area_x', 'Cn_i', 'Cn_avgSens',
                                                   'Cn_avgSpec', 'C_local', 'Ux_count', 'Uy_count']]
            #endif
            if not quiet:
                showCmeasures(groupIndex+1, C_i, C_area_x, C_area_y, Cn_i, Cn_avgSpec, Cn_avgSens,
                              Ux_count, Uy_count, whole_area, vertical_stripe_area, horizontal_stripe_area)
            #endif
        #endif

        # compute measures related to the concordant partial AUC
        temp_dict = concordant_partial_AUC(partial_fpr, partial_tpr, quiet)
        measure_dict.update(temp_dict)
        AUC_i, pAUC, pAUCx, AUCn_i, pAUCn, pAUCxn = \
            [measure_dict[key] for key in ['AUC_i', 'pAUC', 'pAUCx', 'AUCn_i', 'pAUCn', 'pAUCxn']]
        if not quiet:
            showAUCmeasures(groupIndex+1, AUC_i, pAUC, pAUCx, AUCn_i, pAUCn, pAUCxn)
        #endif

        # for a group spanning the whole ROC plot, show whole measures
        if self.groups[groupIndex][0] == 0 and self.groups[groupIndex][1] == 1:
            # AUC was already computed by the constructor
            # note: avoid: AUC = metrics.auc(ffpr, ftpr)
            # it sometimes gives an error: ValueError: x is neither increasing nor decreasing
            # use trapz instead...
            if forFolds:
                AUC_full = np.trapz(self.mean_tpr, self.mean_fpr)
                temp_dict = dict(AUC_full=AUC_full)
            else:
                AUC_full  = np.trapz(self.full_tpr, self.full_fpr)
                AUC_macro = metrics.roc_auc_score(self.newlabels, self.predicted_scores)  # macro is default
                AUC_micro = metrics.roc_auc_score(self.newlabels, self.predicted_scores, average='micro')
                AUPRC     = metrics.average_precision_score(self.newlabels, self.predicted_scores)  # macro is default
                temp_dict = dict(AUC=self.AUC, AUC_full=AUC_full, AUC_macro=AUC_macro, AUC_micro=AUC_micro, AUPRC=AUPRC)

                if not quiet:
                    showWholeAUCmeasures(self.AUC, AUC_full, AUC_macro, AUC_micro, AUPRC)
                #endif
            #endif
            measure_dict.update(temp_dict)
        #endif

        # discrete partial measures next
        if forFolds:
            N = self.foldsNPclassRatio / (1 + self.foldsNPclassRatio)
            P = 1 - N
            prevalence = P / (P+N)
        else:
            # if populationPrevalence not set, then use the whole sample to derive an estimate
            if self.populationPrevalence == None:
                prevalence = len(posScores) / (len(posScores) + len(negScores))
            else:
                prevalence = self.populationPrevalence
            # endif
            N = float(sum(negWeights))  # group sample
            P = float(sum(posWeights))  # group sample
        #endif
        temp_dict = discrete_partial_roc_measures(partial_fpr, partial_tpr, N, P, prevalence)
        measure_dict.update(temp_dict)
        if not quiet:
            showAllMeasures = True
            showDiscretePartialAUCmeasures(measure_dict, showAllMeasures)
        #endif

        # show PAI result only if the group ends at FPR == 0
        if self.groupAxis == 'FPR' and self.groups[groupIndex][0] == 1:
            PAI = partial_area_index_proxy(partial_fpr, partial_fpr, quiet)
            measure_dict.update(dict(PAI=PAI))
            if not quiet:
                print(f"{'PAI':16s} = {PAI:0.4f}  only applies to full or last range")
            # endif
        # endif

        if not forFolds:
            # check for expected equalities
            ep    = 1 * (10 ** -12)
            pass1 = areEpsilonEqual(C_i,  AUC_i,   'C_i',  'AUC_i', ep, quiet)
            pass2 = areEpsilonEqual(Cn_i, AUCn_i, 'Cn_i', 'AUCn_i', ep, quiet)

            if matchedIndices[0] != 'NA' and matchedIndices[1] != 'NA':
                pass3 = areEpsilonEqual(measure_dict['bAvgA'],   AUCn_i, 'bAvgA',   'AUCn_i', ep, quiet)
                pass4 = areEpsilonEqual(measure_dict['avgSens'], pAUCn,  'avgSens', 'pAUCn',  ep, quiet)
            # endif

            pass5 = areEpsilonEqual(Cn_avgSens, pAUCn, 'Cny_i', 'pAUCn', ep, quiet)

            if matchedIndices[0] != 'NA' and matchedIndices[1] != 'NA':
                pass6 = areEpsilonEqual(measure_dict['avgSpec'], pAUCxn, 'avgSpec', 'pAUCxn', ep, quiet)
            # endif

            pass7 = areEpsilonEqual(Cn_avgSpec, pAUCxn, 'Cnx_i', 'pAUCxn', ep, quiet)

            if matchedIndices[0] != 'NA' and matchedIndices[1] != 'NA':
                iterationPassed = pass1 and pass2 and pass3 and pass4 and pass5 and pass6 and pass7
            else:
                iterationPassed = pass1 and pass2 and pass5 and pass7
            # endif

            return iterationPassed, measure_dict
        else:
            return True, measure_dict
        #endif
    #enddef

    def analyze(self):
        from Helpers.DeepROCFunctions import areEpsilonEqual

        measure_dict = []
        if self.groupsArePerfectCoveringSet:
            Ci_sum, AUCi_sum, pAUC_sum, pAUCx_sum = [0, 0, 0, 0]
            allIterationsPassed = True
        # endif
        numgroups = len(self.groups)
        for i in range(0, numgroups):
            print(f'\nGroup {i + 1}:')
            iterationPassed, iteration_dict = self.analyzeGroup(i, showData=True, forFolds=False, quiet=False)
            measure_dict = measure_dict + [iteration_dict]
            # add up parts as you go
            if self.groupsArePerfectCoveringSet:
                allIterationsPassed = allIterationsPassed and iterationPassed
                Ci_sum              = Ci_sum     + measure_dict[i]['C_i']
                AUCi_sum            = AUCi_sum   + measure_dict[i]['AUC_i']
                pAUC_sum            = pAUC_sum   + measure_dict[i]['pAUC']
                pAUCx_sum           = pAUCx_sum  + measure_dict[i]['pAUCx']
            # endif
        # endfor

        # code to check for PASS here
        if self.groupsArePerfectCoveringSet:
            ep = 1 * (10 ** -12)
            quietFalse = False
            print(' ')
            pass1 = areEpsilonEqual(Ci_sum,    self.C,   'C_i_sum',   'C',   ep, quietFalse)
            pass2 = areEpsilonEqual(AUCi_sum,  self.AUC, 'AUC_i_sum', 'AUC', ep, quietFalse)
            pass3 = areEpsilonEqual(pAUC_sum,  self.AUC, 'pAUC_sum',  'AUC', ep, quietFalse)
            pass4 = areEpsilonEqual(pAUCx_sum, self.AUC, 'pAUCx_sum', 'AUC', ep, quietFalse)

            allPassed = allIterationsPassed and pass1 and pass2 and pass3 and pass4
            if allPassed:
                print(f"\nAll results passed.")
            else:
                print(f"\nSome results did not match (failed).")
            # endif
        # endif
    #enddef

    def setFoldsNPclassRatio(self, foldsNPclassRatio):
        self.foldsNPclassRatio = foldsNPclassRatio
    #enddef

    def setNPclassRatio(self, NPclassRatio):
        self.NPclassRatio = NPclassRatio
    #enddef

    def setPriorPoint(self, priorPoint):
        self.priorPoint = priorPoint
    #enddef

    # plot() is in the superclass FullROC

    def plotGroupForFolds(self, plotTitle, groupIndex, foldsNPclassRatio, showError=False, showThresholds=True,
                          showOptimalROCpoints=True, costs=None, saveFileName=None, numShowThresh=20,
                          showPlot=True, labelThresh=True, full_fpr_tpr=True):
        '''plotGroupForFolds shows a mean ROC plot for a contiguous group (any group except by instance).'''
        forFolds = True
        return self.plotGroupInternalLogic(plotTitle, groupIndex, foldsNPclassRatio,
                                       forFolds=forFolds, showError=showError,
                                       showThresholds=showThresholds, showOptimalROCpoints=showOptimalROCpoints,
                                       costs=costs, saveFileName=saveFileName, numShowThresh=numShowThresh,
                                       showPlot=showPlot, labelThresh=labelThresh, full_fpr_tpr=full_fpr_tpr)

    #enddef

    def plotGroup(self, plotTitle, groupIndex, showError=False, showThresholds=True, showOptimalROCpoints=True,
                  costs=None, saveFileName=None, numShowThresh=20, showPlot=True, labelThresh=True,
                  full_fpr_tpr=True):
        '''plotGroup shows an ROC plot for a contiguous groups (any group except by instance).'''
        forFolds = False
        foldsNPclassRatio = None
        return self.plotGroupInternalLogic(plotTitle, groupIndex, foldsNPclassRatio, forFolds=forFolds,
                                    showError=showError,
                                    showThresholds=showThresholds, showOptimalROCpoints=showOptimalROCpoints,
                                    costs=costs, saveFileName=saveFileName, numShowThresh=numShowThresh,
                                    showPlot=showPlot, labelThresh=labelThresh, full_fpr_tpr=full_fpr_tpr)
    #enddef

    def plotGroupInternalLogic(self, plotTitle, groupIndex, foldsNPclassRatio, forFolds=False, showError=False,
                               showThresholds=True, showOptimalROCpoints=True, costs=None, saveFileName=None,
                               numShowThresh=20, showPlot=True, labelThresh=True, full_fpr_tpr=True):
        import matplotlib.pyplot as plt
        import math
        import numpy as np
        from Helpers.ROCPlot      import addPointsAndLabels
        from Helpers.ROCPlot      import plotOptimalPointWithThreshold
        from Helpers.ROCFunctions import getSlopeOrSkew
        from Helpers.ROCFunctions import optimal_ROC_point_indices
        from Helpers.DeepROCPlot  import plotPartialArea
        from Helpers.ROCFunctions import plot_major_diagonal

        # plot full ROC data for whole and partial curve
        # with thresholds labeled and the Metz optimal ROC point(s) indicated

        if forFolds and self.groupAxis == 'Threshold':
            ValueError('Mean ROC plots do not have Thresholds, use Percentiles instead.')
        elif not forFolds and self.groupAxis == 'Threshold':
            ValueError('Plot not available for groups by Threshold at this time.')
        # endif

        showInterimPlot = False
        if forFolds:
            fig, ax = self.plot_folds(plotTitle, saveFileName=saveFileName, showPlot=showInterimPlot)
        else:
            fig, ax = self.plot(plotTitle, showThresholds, showOptimalROCpoints, costs, saveFileName,
                                numShowThresh, showInterimPlot, labelThresh, full_fpr_tpr)
        #endif

        if self.groups[groupIndex][0] == 0:
            rocRuleLeft  = 'SW'
            rocRuleRight = 'NE'
        else:
            rocRuleLeft  = 'NE'
            rocRuleRight = 'NE'
        # endif

        if self.groupAxis == 'TPR' or self.groupAxis == 'FPR':
            quiet = True
            if forFolds:
                fpr = self.mean_fpr
                tpr = self.mean_tpr
                thresholds = np.ones(self.mean_fpr.shape)  # dummy thresholds
            else:
                fpr = self.full_fpr
                tpr = self.full_tpr
                thresholds = self.full_thresholds
            #endif

            partial_fpr, partial_tpr, groupByOtherAxis, groupByThreshold, matchedIndices, approxIndices \
                = self.getGroupForAUC(fpr, tpr, thresholds, self.groupAxis, self.groups[groupIndex],
                                      rocRuleLeft, rocRuleRight, quiet)

            # add fills for partial areas (the fills clobber any points or labels)
            plotPartialArea(partial_fpr, partial_tpr, showError)

            if not forFolds:
                fancyLabel = True
                addPointsAndLabels(fpr, tpr, numShowThresh, thresholds, fancyLabel)
            #endif
        # endif

        if showOptimalROCpoints:
            if forFolds:
                slopeOrSkew = getSlopeOrSkew(foldsNPclassRatio, costs)
                N   = foldsNPclassRatio / (1 + foldsNPclassRatio)
                P   = 1 - N
                prevalence  = P / (P + N)
                opt_indices = optimal_ROC_point_indices(fpr, tpr, slopeOrSkew)
                plt.scatter(fpr[opt_indices], tpr[opt_indices], s=40, marker='o', alpha=1, facecolors='w', lw=2,
                            edgecolors='r')
            else:
                P = int(sum(self.full_newlabels))
                N = len(self.full_newlabels) - P
                prevalence  = P / (P + N)
                slopeOrSkew = getSlopeOrSkew(N / P, costs)
                opt_indices = optimal_ROC_point_indices(fpr, tpr, slopeOrSkew)

                if not math.isinf(thresholds[0]):
                    maxThreshold = thresholds[0]  # if first (max) thresh is not infinite,
                                                  # then use it for label
                else:
                    maxThreshold = thresholds[1]  # otherwise, use the next label which is finite
                # endif
                fancyLabel = True
                plotOptimalPointWithThreshold(fpr[opt_indices], tpr[opt_indices], thresholds[opt_indices],
                                              maxThreshold, fancyLabel)
            #endif
        # endif

        plot_major_diagonal()

        #if self.__class__.__name__ == 'BayesianROC':
        #    from Helpers.BayesianROCFunctions import plot_bayesian_iso_line
        #    plot_bayesian_iso_line(prevalence, costs, self.BayesianPrior)
        ##endif

        plt.xlim(-0.01, 1.0)
        plt.ylim(0.0, 1.01)

        if showPlot:
            plt.show()
        # endif

        # modeShort = mode[:-3]  # training -> train, testing -> test
        # fig.savefig(f'output/ROC_{modeShort}_{testNum}-{index}.png')

        return fig, ax
    #enddef

    def plotGroupInConcordanceMatrix(self, plotTitle, showThresholds=True, showOptimalROCpoints=True,
                                      costs=None, saveFileName=None, numShowThresholds=20, showPlot=True):
        '''plots a Concordance Matrix for contiguous groups (any groups except by instance).'''
        from ConcordanceMatrixPlot import ConcordanceMatrixPlot

        if self.groupAxis == 'Instances':
            SystemError('The ConcordanceMatrixPlot is possible, but not available for groups by instance.')
        aCMplot   = ConcordanceMatrixPlot(ROCdata=self)
        aCMplot.setGroupsBy(groupAxis=self.groupAxis, groups=self.groups, groupByClosestInstance=False)
        showError = False
        fig, ax   = aCMplot.plotGroups(plotTitle, showError, showThresholds, showOptimalROCpoints,
                                       costs, saveFileName, numShowThresholds, showPlot)
        return fig, ax
    #enddef

    # def plotPoints(self):
    #     '''plotPoints shows or overlays ROC plots with specific points labelled.'''
    #     # to be completed...
    #     return False
    #enddef

    def getGroupForAUC(self, ffpr, ftpr, fthresh, rangeAxis1, rangeEndpoints1,
                       rocRuleLeft, rocRuleRight, quiet):
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
        import numpy as np
        from Helpers.DeepROCFunctions import rocErrorCheck
        from Helpers.DeepROCInterpolation import get_Match_or_Interpolation_Points
        from Helpers.DeepROCInterpolation import interpolateROC
        #
        FPR, TPR, NE, SW = ('FPR', 'TPR', 'NE', 'SW')
        # error checks
        rocErrorCheck(ffpr, ftpr, fthresh, rangeEndpoints1, rangeAxis1, rocRuleLeft, rocRuleRight)

        pfpr_np = np.array(ffpr)
        ptpr_np = np.array(ftpr)
        n = len(ffpr)

        if rangeAxis1 == FPR:
            rstat = ffpr.copy()  # range statistic
            ostat = ftpr.copy()  # other statistic
        else:  # rangeAxis1 == TPR:
            rstat = ftpr.copy()  # range statistic
            ostat = ffpr.copy()  # other statistic
        # endif

        rangeIndices0   = ['NA', 'NA']  # initialize to detectable nonsense value
        approxIndices0  = ['NA', 'NA']  # initialize to detectable nonsense value
        rangeEndpoints0 = ['NA', 'NA']  # initialize to detectable nonsense value
        rangeEndpoints2 = ['NA', 'NA']  # initialize to detectable nonsense value
        # we need to process the endpoints in right to left order [1, 0] so that
        # when we delete points from pfpr and ptpr, the indices on the left still
        # make sense even if we have changed the indices on the right.  In the
        # other direction both indices are affected.
        indices_reversed = [1, 0]
        rangeEndpoints1_reversed = [rangeEndpoints1[1], rangeEndpoints1[0]]
        rocRules_reversed = [rocRuleRight, rocRuleLeft]
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
                # endif
                if not quiet:
                    print(f'Interpolating {rangeAxis1}[{i}] between {rstat[ixL]:0.3f} and {rstat[ixR]:0.3f}')
                #   with a newly interpolated point at rangeEndpoints0
                if rangeAxis1 == FPR:
                    if i == 1:  # right/top
                        pfpr_np = np.delete(pfpr_np, np.arange(ixR, n))
                        ptpr_np = np.delete(ptpr_np, np.arange(ixR, n))
                        pfpr_np = np.append(pfpr_np, endpoint)
                        ptpr_np = np.append(ptpr_np, rangeEndpoints2[i])
                    elif i == 0:  # left/bottom
                        pfpr_np = np.delete(pfpr_np, np.arange(0, ixL))
                        ptpr_np = np.delete(ptpr_np, np.arange(0, ixL))
                        pfpr_np = np.insert(pfpr_np, 0, endpoint)
                        ptpr_np = np.insert(ptpr_np, 0, rangeEndpoints2[i])
                    # endif
                else:  # rangeAxis1 == TPR:
                    if i == 1:  # right/top
                        pfpr_np = np.delete(pfpr_np, np.arange(ixR, n))
                        ptpr_np = np.delete(ptpr_np, np.arange(ixR, n))
                        pfpr_np = np.append(pfpr_np, rangeEndpoints2[i])
                        ptpr_np = np.append(ptpr_np, endpoint)
                    elif i == 0:  # left/bottom
                        pfpr_np = np.delete(pfpr_np, np.arange(0, ixL))
                        ptpr_np = np.delete(ptpr_np, np.arange(0, ixL))
                        pfpr_np = np.insert(pfpr_np, 0, rangeEndpoints2[i])
                        ptpr_np = np.insert(ptpr_np, 0, endpoint)
                    # endif
                # endif
            else:  # found one or more indices in ix that match endpoint
                # use rules to choose which of multiple matching points to use
                # (this logic also works for a single matching point)
                if rocRule == SW:  # take earliest point
                    ix_to_use = ix[0]
                else:  # rocRule == NE: # take last point
                    ix_to_use = ix[-1]
                # endif
                rangeIndices0[i] = ix_to_use
                rangeEndpoints0[i] = fthresh[ix_to_use]
                rangeEndpoints2[i] = ostat[ix_to_use]
                if i == 1:  # right/top
                    if ix_to_use < n - 1:  # if not last  instance then truncate right part
                        pfpr_np = np.delete(pfpr_np, np.arange(ix_to_use + 1, n))
                        ptpr_np = np.delete(ptpr_np, np.arange(ix_to_use + 1, n))
                    # endif
                elif i == 0:  # left/bottom
                    if ix_to_use > 0:  # if not first instance then truncate left part
                        pfpr_np = np.delete(pfpr_np, np.arange(0, ix_to_use))
                        ptpr_np = np.delete(ptpr_np, np.arange(0, ix_to_use))
                    # endif
                # endif
            # endif
        # endfor
        pfpr = pfpr_np.tolist()
        ptpr = ptpr_np.tolist()
        return pfpr, ptpr, rangeEndpoints2, rangeEndpoints0, rangeIndices0, approxIndices0
    # enddef

    def getGroupForC(self, ffpr, ftpr, fthresh, fSlopeFactor, scores, labels, posclass, xrange, yrange):
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
        import numpy as np
        import math
        from   Helpers.ROCFunctions import range01Check
        from   Helpers.DeepROCFunctions import sortScoresAndLabels4

        # check inputs for errors
        n     = len(fthresh)
        if n != len(ffpr) and n != len(ftpr):
            return "Error, fpr and tpr and thresh must have the same length"
        # cErrorCheck(cRuleLeft, cRuleRight)
        range01Check(xrange)
        range01Check(yrange)

        # we will obtain the positive and negative instances separately in sorted
        # order from highest value near ROC (0,0) to lowest value at the furthest
        # value therefrom.  See the concordance matrix in Carrington et al, 2020.

        self.predicted_scores, self.newlabels, self.labels, self.sorted_full_slope_factors = \
            sortScoresAndLabels4(self.predicted_scores, self.newlabels, self.labels, self.full_slope_factor)

        # get the positive instances (their scores) along the TPR axis
        # also get the height travelled by each instance in a sloped segment
        posIdx     = np.argwhere(np.array(self.newlabels) == 1).flatten()
        posScores  = []
        posHeights = []
        for ix in posIdx:
            posScores  = posScores  + [self.predicted_scores[ix]]
            posHeights = posHeights + [self.sorted_full_slope_factors[ix]]
        # endfor

        # get the negative instances (their scores) along the FPR axis
        # also get the width travelled by each instance in a sloped segment
        negIdx    = np.argwhere(np.array(self.newlabels) == 0).flatten()
        negScores = []
        negWidths = []
        for ix in negIdx:
            negScores = negScores + [self.predicted_scores[ix]]
            negWidths = negWidths + [1 - self.sorted_full_slope_factors[ix]]
        # endfor

        # initialize pos and neg indices and weights
        ptotal     = len(posScores)
        ntotal     = len(negScores)
        posWeights = np.ones((ptotal, 1), float)
        negWeights = np.ones((ntotal, 1), float)
        posWeight  = np.zeros((2, 1),     float)
        negWeight  = np.zeros((2, 1),     float)
        negIndexC  = np.zeros((2, 1),     int)  # or np.int, it doesn't matter which
        posIndexC  = np.zeros((2, 1),     int)

        # get index of the left boundary instance and the updated weights that
        # interpolate that instance (where weight is left-to-right thinking)
        def interpolateAxis(axisValue, numAxisInstances):
            rawIndex        = axisValue * numAxisInstances  # rawIndex has a decimal value, e.g. 1.2
            indexBefore     = math.floor(rawIndex)  # 0-based index
            if indexBefore == rawIndex and indexBefore > 0:
                indexBefore = indexBefore - 1
            weight = rawIndex - indexBefore
            return indexBefore, weight
        # enddef

        negIndexC[0], negWeight[0] = interpolateAxis(xrange[0], ntotal)
        negIndexC[1], negWeight[1] = interpolateAxis(xrange[1], ntotal)
        if negIndexC[0] == negIndexC[1]:
            # this case is complicated: we are interpolating twice, both the
            # top and bottom boundaries in the same instance
            negWeights[negIndexC[0]] = negWeight[1] - negWeight[0]  # set weight at boundary
        else:
            # interpolated weight is based on left to right thinking (0 at left, 1 at right)
            # but the weight we need for a left boundary is opposite, so
            negWeight0_reversed      = 1 - negWeight[0]
            negWeights[negIndexC[0]] = negWeight0_reversed  # set weight at boundary
            negWeights[negIndexC[1]] = negWeight[1]         # set weight at boundary
        # endif
        # cast single element numpy array to int, for range/slice to work
        negWeights[0: int(negIndexC[0])] = 0           # zeroize left of left boundary
        negWeights[int(negIndexC[1]) + 1: ntotal] = 0  # zeroize right of right boundary

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
            posWeights[posIndexC[1]] = posWeight[1]  # set weight at boundary
        # endif
        # cast single element numpy array to int, for range/slice to work
        posWeights[0: int(posIndexC[0])] = 0  # zeroize below bottom boundary
        posWeights[int(posIndexC[1]) + 1: ptotal] = 0  # zeroize above top boundary

        return negIndexC, posIndexC, negScores, posScores, negWeights, posWeights, negWidths, posHeights
    # enddef

    def __str__(self):
        '''This method prints the object as a string of its content re 
           predicted scores, labels, full fpr, full tpr, full thresholds.'''
        
        if self.predicted_scores is None and self.full_fpr and self.full_tpr and self.full_thresholds:
            rocdata = f'score, label, fullfpr, fulltpr, fullthr\n'
            for c, d, e in zip(self.full_fpr, self.full_tpr, self.full_thresholds):
                rocdata = rocdata + f'{c:0.3f}  , {d:0.3f}  , {e:0.3f}\n'
            #endfor
        elif self.predicted_scores is None and self.full_fpr == None:
            rocdata = f'score, label, fullfpr, fulltpr, fullthr\n'
            rocdata = rocdata + \
             f'{self.predicted_scores}, {self.labels}, {self.full_fpr}, {self.full_tpr}, {self.full_thresholds}'
        else:
            rocdata = f'score, label, fullfpr, fulltpr, fullthr\n'
            for a, b, c, d, e in zip(self.predicted_scores, self.labels,
                                     self.full_fpr, self.full_tpr, self.full_thresholds):
                rocdata = rocdata + f'{a:0.3f}, {b:<5d}, {c:0.3f}  , {d:0.3f}  , {e:0.3f}\n'
            #endfor
        #endif
        rocdata = rocdata + f'\nfpr_fold, tpr_fold, thresholds_fold\n'
        rocdata = rocdata + \
                  f'{self.fpr_fold}\n{self.tpr_fold}\n{self.thresholds_fold}\n'
        return rocdata
    #enddef 