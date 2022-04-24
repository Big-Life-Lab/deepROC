#!/usr/bin/env python
# -*- coding: latin-1 -*-
# SimpleROC.py
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

class SimpleROC(object):
    
    def __init__(self, predicted_scores=None, labels=None, poslabel=None, quiet=False):
        '''SimpleROC constructor. If predicted_scores and labels are
           empty then it returns an empty object.'''
        from Helpers.ROCFunctions import checkFixLabels
        from Helpers.ROCFunctions import C_statistic
        from sklearn import metrics

        if predicted_scores is not None and labels is not None:
            self.predicted_scores               = predicted_scores
            self.labels                         = labels
            self.poslabel                       = poslabel
            self.newlabels, self.newposlabel    = checkFixLabels(labels, poslabel)
            self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.newlabels,
                                                                    self.predicted_scores,
                                                                    pos_label=self.newposlabel)
            self.AUC                            = metrics.roc_auc_score(self.newlabels,
                                                                        self.predicted_scores)
            self.C                              = C_statistic(self.predicted_scores, self.newlabels)
            self.optimalpoints                                 = None
            self.nextfold                                      = 0
            self.fpr_fold, self.tpr_fold, self.thresholds_fold = [], [], []
            self.meanAUC, self.stdAUC, self.AUCofMeanROC       = None, None, None
            self.AUClowCI, self.AUChighCI, self.AUCs           = None, None, None
            self.mean_fpr, self.mean_tpr, self.std_tpr         = None, None, None
        else:
            self.predicted_scores, self.labels                 = None, None
            self.poslabel, self.newlabels, self.newposlabel    = None, None, None
            self.fpr, self.tpr, self.thresholds                = None, None, None
            self.AUC, self.C, self.optimalpoints               = None, None, None
            self.nextfold                                      = 0
            self.fpr_fold, self.tpr_fold, self.thresholds_fold = [], [], []
            self.meanAUC, self.stdAUC, self.AUCofMeanROC       = None, None, None
            self.AUClowCI, self.AUChighCI, self.AUCs           = None, None, None
            self.mean_fpr, self.mean_tpr, self.std_tpr         = None, None, None
        #endif
    #enddef

    def get_fpr_tpr(self):
        from sklearn.metrics import roc_curve
        ''' Computes and returns the AUC or AUROC (a continuous measure)'''
        if self.fpr is not None and self.tpr is not None:
            return self.fpr, self.tpr
        #endif
        if self.predicted_scores is not None and self.newlabels is not None:
            self.fpr, self.tpr, self.thresholds = roc_curve(self.newlabels, self.predicted_scores)
            return self.fpr, self.tpr
        else:
            SystemError('Predicted scores and labels, or FPR and TPR, are required to compute the AUC.')
        #endif
    #enddef

    def getAUC(self):
        ''' Computes and returns the AUC or AUROC (a continuous measure)'''
        import sklearn.metrics as metrics

        if self.predicted_scores is None and self.newlabels is None:
            if self.fpr is not None and self.tpr is not None:
                self.AUC = metrics.auc(self.fpr, self.tpr)
            else:
                SystemError('Predicted scores and labels, or FPR and TPR, are required to ' +
                            'compute the AUC.')
            #endif
        else:  # self.predicted_scores and self.newlabels are populated
            self.AUC     = metrics.roc_auc_score(self.newlabels, self.predicted_scores)
        #endif
        return self.AUC
    #enddef

    def getC(self):
        ''' Computes and returns the C statistic (a discrete measure)'''
        from Helpers.ROCFunctions import C_statistic

        if self.predicted_scores is None or self.newlabels is None:
            SystemError('Actual labels and predicted scores are required to compute the C statistic.')
        else:
            self.C = C_statistic(self.predicted_scores, self.newlabels)
            return self.C
        #endif
    #enddef
    
    def get(self):
        '''get returns the arrays for predicted_scores, labels, fpr, tpr, thresholds.'''
        import sklearn.metrics as metrics

        if self.predicted_scores is not None and self.newlabels is not None:            
            if self.fpr is None and self.tpr is None:
                self.fpr, self.tpr, self.thresholds = metrics.roc_curve(
                                                        self.newlabels, 
                                                        self.predicted_scores, 
                                                        pos_label=self.newposlabel)
            #endif
        #endif

        if self.__class__.__name__ == 'SimpleROC':  # as opposed to a subclass
            msg = 'sklearn metrics.roc_curve sets the highest threshold ' + \
                  'to max+1, when it should/may be any threshold above max: (max, infinity].'
            print(f'Warning from get(): {msg}')
        #endif

        # note: we return newlabels (not labels) in the following
        return self.predicted_scores, self.newlabels, self.fpr, self.tpr, self.thresholds
    #enddef

    def plot(self, plotTitle, showThresholds=True, showOptimalROCpoints=True, costs=None,
             saveFileName=None, numShowThresh=30, showPlot=True, labelThresh=False, full_fpr_tpr=False):
        '''plot provides an ROC plot with full data (including a point for each tie), and
           optional labels for threshold percentiles or thresholds, and optional optimal ROC points.'''
        '''plotWholeROC plots the whole curve with thresholds labeled and the Metz optimal ROC point(s) indicated'''
        from   Helpers.ROCPlot      import plotROC
        from   Helpers.ROCPlot      import plotOptimalPointWithThreshold
        from   Helpers.ROCFunctions import getSlopeOrSkew
        from   Helpers.ROCFunctions import plot_major_diagonal
        from   Helpers.ROCFunctions import optimal_ROC_point_indices
        import matplotlib.pyplot as plt
        import math

        if self.__class__.__name__ == 'SimpleROC':  # as opposed to a subclass
            msg = 'sklearn metrics.roc_curve sets the highest threshold ' + \
                  'to max+1, when it should/may be any threshold above max: (max, infinity].'
            print(f'Warning from plot(): {msg}')
        #endif

        if self.__class__.__name__ == 'SimpleROC' or not full_fpr_tpr:
            fpr         = self.fpr
            tpr         = self.tpr
            thresholds  = self.thresholds
            newlabels   = self.newlabels
            newposlabel = self.newposlabel
        else:
            fpr         = self.full_fpr
            tpr         = self.full_tpr
            thresholds  = self.full_thresholds
            newlabels   = self.full_newlabels
        #endif

        newposlabel = self.newposlabel  # [sic] there is no full version for this

        fig, ax     = plotROC(fpr, tpr, plotTitle, numShowThresh, thresholds, labelThresh)

        if showOptimalROCpoints:
            # get optimal points here...
            P = int(sum(newlabels))
            N = len(newlabels) - P
            slopeOrSkew = getSlopeOrSkew(N/P, costs)

            optimalpoints  = optimal_ROC_point_indices(fpr, tpr, slopeOrSkew)
            fpr_opt        = fpr[optimalpoints]
            tpr_opt        = tpr[optimalpoints]
            thresholds_opt = thresholds[optimalpoints]

            # for plotOpt...
            if not math.isinf(thresholds[0]):
                maxThreshold = thresholds[0]  # if first (max) thresh is not infinite, then use it for label
            else:
                maxThreshold = thresholds[1]  # otherwise, use the next label which should be finite
            # endif

            plotOptimalPointWithThreshold(fpr_opt, tpr_opt, thresholds_opt, maxThreshold, labelThresh)  # add the optimal ROC points
        # endif

        plot_major_diagonal()

        if showPlot:
            plt.show()
        #modeShort = mode[:-3]  # training -> train, testing -> test
        #fig.savefig(f'output/ROC_{modeShort}_{testNum}-{index}.png')

        if saveFileName is not None:
            fig.savefig(saveFileName)

        return fig, ax
    #enddef

    def plot_folds(self, plotTitle, saveFileName=None, showPlot=True):
        from Helpers.ROCPlot import plotSimpleROC
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import auc

        if self.nextfold <= 2:
            ValueError('Multiple folds have not been set.')
        #endif

        # some ideas borrowed from:
        # https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/plot_roc_crossval.html
        # https://stackoverflow.com/questions/57708023/plotting-the-roc-curve-of-k-fold-cross-validation
        # but improved here, re the (0,0) ROC point
        fig      = plt.figure()
        ax       = fig.add_subplot(1, 1, 1)
        mean_fpr = np.linspace(0, 1, 200)
        mean_fpr = np.insert(mean_fpr, 0, 0)  # insert an extra 0 at the beginning
        mean_fpr = np.append(mean_fpr, 1)     # insert an extra 1 at the end
        tprs     = []
        aucs     = []
        for i in range(0, self.nextfold):
            tprs.append(np.interp(mean_fpr, self.fpr_fold[i], self.tpr_fold[i]))
            # interestingly interp, for multiple values of y at x=0, correctly
            # takes the highest value. one simply needs to insert a point (0,0)
            # at the beginning (the extra 0 previously inserted, so overwrite it)
            tprs[i][0]  = 0.0
            tprs[i][-1] = 1.0
            aucs.append(auc(self.fpr_fold[i], self.tpr_fold[i]))
            plt.plot(self.fpr_fold[i], self.tpr_fold[i], lw=2, alpha=0.3,
                     label=f'Fold {i+1}, AUC={aucs[i]:0.2f}')
        #endfor

        # add major diagonal
        x = np.linspace(0, 1, 3)
        plt.plot(x, x, linestyle=':', color='black')  # default linewidth is 1.5
        plt.plot(x, x, linestyle='-', color='black', linewidth=0.25)
        # the above thin (not quite visible) solid line, stops color fills from passing through it

        # add major diagonal
        self.mean_fpr     = mean_fpr
        self.mean_tpr     = np.mean(tprs, axis=0)
        self.AUCofMeanROC = auc(self.mean_fpr, self.mean_tpr)
        self.meanAUC      = np.mean(aucs)
        self.stdAUC       = np.std(aucs)
        self.AUCs         = aucs
        self.AUChighCI    = np.minimum(self.meanAUC + (2 * self.stdAUC), 1)
        self.AUClowCI     = np.maximum(self.meanAUC - (2 * self.stdAUC), 0)
        plotSimpleROC(self.mean_fpr, self.mean_tpr, plotTitle)
        plt.legend()
        # print(f'Mean ROC, AUC={self.meanAUC:0.3f} +/- {self.stdAUC:0.3f}')
        # print(f'AUC of mean ROC is {self.AUCofMeanROC:0.3f}')

        self.std_tpr = np.std(tprs, axis=0)
        tpr_upper    = np.minimum(self.mean_tpr + (2*self.std_tpr), 1)
        tpr_lower    = np.maximum(self.mean_tpr - (2*self.std_tpr), 0)
        plt.fill_between(self.mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=.2,
                         label=r'Mean ROC $\pm$2 stddev.')
        if showPlot:
            plt.show()
        #endif
        if saveFileName is not None:
            fig.savefig(saveFileName)
        #endif
        return fig, ax
    #enddef

    def getAUCofMeanROC(self):
        import numpy as np
        from sklearn.metrics import auc

        if self.nextfold <= 2:
            ValueError('Multiple folds have not been set.')
        # endif

        if self.AUCofMeanROC != None:
            return self.AUCofMeanROC
        else:
            mean_fpr = np.linspace(0, 1, 200)
            mean_fpr = np.insert(mean_fpr, 0, 0)  # insert an extra 0 at the beginning
            mean_fpr = np.append(mean_fpr, 1)  # insert an extra 1 at the end
            tprs = []
            aucs = []
            for i in range(0, self.nextfold):
                tprs.append(np.interp(mean_fpr, self.fpr_fold[i], self.tpr_fold[i]))
                # interestingly interp, for multiple values of y at x=0, correctly
                # takes the highest value. one simply needs to insert a point (0,0)
                # at the beginning (the extra 0 previously inserted, so overwrite it)
                tprs[i][0] = 0.0
                tprs[i][-1] = 1.0
                aucs.append(auc(self.fpr_fold[i], self.tpr_fold[i]))
            # endfor

            mean_tpr          = np.mean(tprs, axis=0)
            self.AUCofMeanROC = auc(mean_fpr, mean_tpr)
            self.meanAUC      = np.mean(aucs)
            self.stdAUC       = np.std(aucs)
            self.AUCs         = aucs
            self.AUChighCI    = np.minimum(self.meanAUC + (2 * self.stdAUC), 1)
            self.AUClowCI     = np.maximum(self.meanAUC - (2 * self.stdAUC), 0)
            return self.AUCofMeanROC
        #endif
    #enddef

    def getMeanAUC_andCI(self):
        import numpy as np
        from sklearn.metrics import auc

        if self.nextfold <= 2:
            ValueError('Multiple folds have not been set.')
        # endif

        if self.meanAUC != None:
            return self.meanAUC, self.AUClowCI, self.AUChighCI, self.AUCs
        else:
            mean_fpr = np.linspace(0, 1, 200)
            mean_fpr = np.insert(mean_fpr, 0, 0)  # insert an extra 0 at the beginning
            mean_fpr = np.append(mean_fpr, 1)  # insert an extra 1 at the end
            tprs = []
            aucs = []
            for i in range(0, self.nextfold):
                tprs.append(np.interp(mean_fpr, self.fpr_fold[i], self.tpr_fold[i]))
                # interestingly interp, for multiple values of y at x=0, correctly
                # takes the highest value. one simply needs to insert a point (0,0)
                # at the beginning (the extra 0 previously inserted, so overwrite it)
                tprs[i][0] = 0.0
                tprs[i][-1] = 1.0
                aucs.append(auc(self.fpr_fold[i], self.tpr_fold[i]))
            # endfor

            mean_tpr          = np.mean(tprs, axis=0)
            self.AUCofMeanROC = auc(mean_fpr, mean_tpr)
            self.meanAUC      = np.mean(aucs)
            self.stdAUC       = np.std(aucs)
            self.AUCs         = aucs
            self.AUChighCI    = np.minimum(self.meanAUC + (2 * self.stdAUC), 1)
            self.AUClowCI     = np.maximum(self.meanAUC - (2 * self.stdAUC), 0)
            return self.meanAUC, self.AUClowCI, self.AUChighCI, self.AUCs
        #endif
    #enddef

    def set_scores_labels(self, predicted_scores=None, labels=None, poslabel=None):
        from Helpers.ROCFunctions import checkFixLabels
        from Helpers.ROCFunctions import C_statistic
        from sklearn import metrics

        if self.predicted_scores is not None and self.newlabels is not None:
            SystemError('predicted_scores and labels are already set.')

        if predicted_scores is None or labels is None:
            SystemError('predicted_scores or labels cannot be empty.')
        else:
            self.predicted_scores               = predicted_scores
            self.labels                         = labels
            self.newlabels, self.newposlabel    = checkFixLabels(labels, poslabel=poslabel)
            self.poslabel                       = poslabel
            self.newlabels, self.newposlabel    = checkFixLabels(labels, poslabel)
            self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.newlabels,
                                                                    self.predicted_scores,
                                                                    pos_label=self.newposlabel)
            self.AUC                            = metrics.roc_auc_score(self.newlabels,
                                                                        self.predicted_scores)
            self.C                              = C_statistic(self.predicted_scores, self.newlabels)
        #endif
    #enddef

    def setFoldsNPclassRatio(self, foldsNPclassRatio):
        self.foldsNPclassRatio = foldsNPclassRatio
    #enddef

    def set_fold(self, fpr=None, tpr=None, threshold=None):
        '''Set ROC data for a fold with fpr and tpr; threshold is optional.'''
        if fpr is None or tpr is None:
            SystemError('fpr or tpr cannot be empty')

        self.fpr_fold.append(fpr)
        self.tpr_fold.append(tpr)
        self.thresholds_fold.append(threshold)
        self.nextfold += 1
    #enddef

    def set_fpr_tpr(self, fpr=None, tpr=None):
        '''The set_fpr_tpr method is allowed if the object is empty.'''
        from sklearn import metrics

        if fpr is None or tpr is None: 
            SystemError('fpr or tpr cannot be empty')
            
        if self.predicted_scores is not None or self.newlabels is not None:
            SystemError('Not allowed to set fpr and tpr ' +
                        'when predicted_scores and labels are already set.')
        
        self.fpr        = fpr 
        self.tpr        = tpr
        self.thresholds = None
        self.AUC        = metrics.auc(self.fpr, self.tpr)
    #enddef

    def __str__(self):
        '''This method prints the object as a string of its content re 
           predicted_scores, labels, fpr, tpr, thresholds.'''

        if self.__class__.__name__ == 'SimpleROC':  # as opposed to a subclass
            msg = 'sklearn metrics.roc_curve sets the highest threshold ' + \
                  'to max+1, when it should/may be any threshold above max: (max, infinity].'
            print(f'Warning from __str__(): {msg}')
        #endif
        rocdata = f'score, label\n'
        for a, b in zip(self.predicted_scores, self.labels):
            rocdata = rocdata + f'{a:0.3f}, {b:<5d}\n'
        #endfor
        rocdata = rocdata + f'\nfpr  , tpr  , thresh\n'
        for c, d, e in zip(self.fpr, self.tpr, self.thresholds):
            rocdata = rocdata + f'{c:0.3f}, {d:0.3f}, {e:0.3f}\n'
        #endfor
        return rocdata
    #enddef

#enddef
