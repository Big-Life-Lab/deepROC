#!/usr/bin/env python
# -*- coding: latin-1 -*-
# deeptest-pypi.py
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

import pandas              as pd
import scipy.io            as sio
from   os.path import splitext
import ntpath
import re
from   deeproc.TestVectors import getTestVector

global quiet, resultFile
quiet       = False
testNum     = 1
useFile     = False
myResults   = ['040', '356', '529', '536', '581', '639', '643']
resultIndex = 3
#resultIndex = 5
groupAxis   = 'FPR'
groups      = [[0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]
#groups     = [[0, 1], [0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]
costs         = dict(cFP=1, cFN=1, cTP=0, cTN=0, costsAreRates=False)  # depends on the dataset
popPrevalence = 0.3  # None uses sample prevalence. population depends on the dataset, 0.3 for LBC data in 536 and 639

def testDeepROC(descr, scores, labels, groupAxis='FPR', groups=[[0.0, 1/3], [1/3, 2/3], [2/3, 1.0]],
                costs=dict(cFP=1, cFN=1, cTP=0, cTN=0, costsAreRates=False), showPlot = True):
    from deeproc.FullROC   import FullROC
    from deeproc.SimpleROC import SimpleROC
    from deeproc.DeepROC   import DeepROC
    from deeproc.ConcordanceMatrixPlot import ConcordanceMatrixPlot

    poslabel      = 1
    numShowThresh = 20

    ######################
    print('SimpleROC:')
    aSimpleROC  = SimpleROC(predicted_scores=scores, labels=labels, poslabel=poslabel)
    print(f'getAUC: {aSimpleROC.getAUC():0.4f}')
    print(f'getC  : {aSimpleROC.getC():0.4f}')
    print(f'get   :\n{aSimpleROC.get()}')
    print('get should produce a warning about the highest threshold')
    if showPlot:
        print('Plot shown.')
        aSimpleROC.plot(plotTitle=f'Simple ROC Plot for Test {descr}',
                        showThresholds=True, showOptimalROCpoints=True, costs=costs,
                        saveFileName=None, numShowThresh=numShowThresh, showPlot=showPlot, labelThresh=True)
    print(f'aSimpleROC:\n{aSimpleROC}')
    # to be do: tests for, set_scores_labels, set_fpr_tpr

    ######################
    print('FullROC:')
    aFullROC = FullROC(predicted_scores=scores, labels=labels, poslabel=poslabel)
    print(f'getAUC: {aFullROC.getAUC():0.4f}')
    print(f'getC  : {aFullROC.getC():0.4f}')
    print(f'get   :\n{aFullROC.get()}')
    if showPlot:
        print('Plot shown.')
        aFullROC.plot(plotTitle=f'Full ROC Plot for Test {descr}',
                      showThresholds=True, showOptimalROCpoints=True, costs=costs,
                      saveFileName=None, numShowThresh=numShowThresh, showPlot=showPlot)
    print(f'aFullROC:\n{aFullROC}')

    ######################
    print('ConcordanceMatrixPlot:')
    aCMplot = ConcordanceMatrixPlot(aFullROC)
    if showPlot:
        print('Plot shown.')
        aCMplot.plot(plotTitle=f'Concordance Matrix for Test {descr}',
                     showThresholds=True, showOptimalROCpoints=True, costs=costs,
                     saveFileName=None, numShowThresholds=numShowThresh, showPlot=showPlot, labelThresh=True)

    ######################
    print('\nDeepROC:')
    aDeepROC = DeepROC(predicted_scores=scores, labels=labels, poslabel=poslabel)
    print(f'getAUC: {aDeepROC.getAUC():0.4f}')
    print(f'getC  : {aDeepROC.getC():0.4f}')
    print(f'get   :\n{aDeepROC.get()}')
    #print('get should NOT produce a warning about the highest threshold, even though the method is inherited')
    if showPlot:
        print('Plot shown.')
        aDeepROC.plot(plotTitle=f'Deep ROC Plot for Test {descr}',
                      showThresholds=True, showOptimalROCpoints=True, costs=costs,
                      saveFileName=None, numShowThresh=numShowThresh, showPlot=showPlot, labelThresh=True, full_fpr_tpr=True)
    print(f'aDeepROC:\n{aDeepROC}')
    numgroups = len(groups)
    print(f'aDeepROC.setGroupsBy: {groupAxis} [ ', end='')
    for i in range(0, numgroups):
        print(f'[{groups[i][0]:0.2f}, {groups[i][1]:0.2f}]', end='')
        if i < numgroups-1:
            print(', ', end='')
    print(' ]')
    aDeepROC.setGroupsBy(groupAxis=groupAxis, groups=groups, groupByClosestInstance=False)

    if showPlot:
        print('aDeepROC.plotGroup:')
        aCMplot2 = ConcordanceMatrixPlot(aDeepROC)
        for i in range(0, numgroups):
            print(f'ROC plot shown for group {i} [{groups[i][0]:0.2f}, {groups[i][1]:0.2f}]')
            aDeepROC.plotGroup(plotTitle=f'Deep ROC Plot for group {i+1}, Test {descr}', groupIndex=i,
                               showError=True, showThresholds=True, showOptimalROCpoints=True, costs=costs,
                               saveFileName=None, numShowThresh=numShowThresh, showPlot=True, labelThresh=True,
                               full_fpr_tpr=True)
            # the following code commented out has bugs...
            # print(f'ConcordanceMatrixPlot shown for group {i} [{groups[i][0]:0.2f}, {groups[i][1]:0.2f}]')
            # aCMplot2.plotGroup(plotTitle=f'Concordance Matrix Plot for group {i+1}, Test {descr}',
            #                    groupIndex=i, showThresholds=True, showOptimalROCpoints=True, costs=costs,
            #                    saveFileName=None, numShowThresholds=numShowThresh, showPlot=showPlot,
            #                    labelThresh=True)
    #endfor

    print('aDeepROC.analyzeGroup:')
    aDeepROC.analyze()
#enddef

def myLoadFile(resultNumString):
    fileName       = f'input/result{resultNumString}.mat'  # a matlab file (or future: csv file) for input
    scoreVariable  = 'yscoreTest'                  # yscoreTest, yscore
    targetVariable = 'ytest'                       # ytest, yhatTest, yhat, ytrain
    return loadMatlabOrCsvFile(fileName, scoreVariable, targetVariable)
#enddef

def loadMatlabOrCsvFile(fileName, scoreVariable, targetVariable):
    if fileName == '':
        SystemError('fileName required.')
    else:  # reduce filename to any 3-digit log number it contains, if possible
        fileNameBase = ntpath.basename(fileName)
        fileNameBase = splitext(fileNameBase)[0]  # remove the extension
        match = re.search(r'\d\d\d', fileNameBase)
        if match:
            fileNum = match.group()
        else:
            fileNum = fileNameBase
        # endif
    #endif

    if fileName[-4:] == '.mat':  # if matlab file input
        try:
            fileContent = sio.loadmat(fileName)  # handle any file not found errors naturally
            scores = fileContent[scoreVariable]
            labels = fileContent[targetVariable]
        except:
            raise ValueError(f'File {fileName} is either not found or is not a matlab file')
        # endtry
    else:  # otherwise assume a CSV file input
        try:
            file_df = pd.read_csv(fileName)
            scores  = file_df[scoreVariable]
            labels  = file_df[targetVariable]
        except:
            raise ValueError(f'File {fileName} is either not found or is not a CSV file')
        #endtry
    #endif

    return scores, labels
#enddef


#######  START OF MAIN LOGIC
if useFile:
    resultNumString = myResults[resultIndex]
    print(f'Test Result {resultNumString}')
    scores, labels  = myLoadFile(resultNumString)
    scores          = scores.flatten()
    labels          = labels.flatten()
    testNum         = int(resultNumString)
else:
    print(f'Test Vector {testNum}')
    scores, labels, dummy_groups, dummy_groupAxis, descr = getTestVector(testNum, noError=False)
#endif
testDeepROC(testNum, scores, labels, groupAxis=groupAxis, groups=groups, costs=costs, showPlot=True)
