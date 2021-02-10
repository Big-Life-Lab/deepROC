# compare15.py
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
#   Original Python version by Andre Carrington, 2021

global total_folds

#import dill
import pickle
import numpy as np
import pandas as pd
import transcript
import scipy.stats as st
from   os.path import splitext

testNum1   = 1
model1     = 'svcMSig'
iteration1 = 2
#iteration1 = 13
measure1   = 'AUC'
#measure1   = 'pAUCn.1'
testNum2   = 1
model2     = 'rf0'
iteration2 = 2
#iteration2 = 32
measure2   = 'AUC'
#measure2   = 'pAUCn.1'
#model2     = 'plogr1'
#iteration2 = 53

iterations = 100
k_folds    = 5
repetition = 2
total_folds= k_folds * repetition

def formatList(alist):
    # Create a format spec for each item in the input `alist`.
    # E.g., each item will be right-adjusted, field width=3.
    format_list = ['{:0.4f}' for item in alist]

    listlen = len(format_list)
    linelen = 5  # start a new line every 5 elements
    addme   = 0
    linewrap= '\n                    ' 
    if listlen > linelen:
        for element in range(linelen, listlen, linelen): # starting at 5 (listlen) add carriage returns
            format_list.insert(element+addme, linewrap)
            addme = addme + 1
        #endfor
    #endif
    # Now join the format specs into a single string:
    # E.g., '{:0.4f}, {:0.4f}, {:0.4f}' if the input list has 3 items.
    s = ', '.join(format_list)

    # Now unpack the input list `alist` into the format string. Done!
    return s.format(*alist)
#enddef

def getVector(measure, iteration, areaMatrix, groupMatrix, areaMeasures, groupMeasures):
    if areaMeasures.count(measure) == 1:
        vector = areaMatrix[:, areaMeasures.index(measure), iteration]
    else:
        parsed  = splitext(measure)
        group   = int(parsed[1][1:])
        measure = parsed[0]
        if groupMeasures.count(measure) == 1:
            vector = groupMatrix[:, group, groupMeasures.index(measure), iteration]
        else:
            raise ValueError(f'Measure {measure}.{group} not found')
        # endif
    # endif
    return vector
# enddef

def analyze(testNum1, model1, iteration1, measure1,
            testNum2, model2, iteration2, measure2, iterations):
    global total_folds

    # load results
    logfn = f'output/compare_{model1}_{testNum1:03d}_{model2}_{testNum2:03d}.txt'
    transcript.start(logfn)
    print(logfn)
    fileHandleS1 = open(f'output/settings_{testNum1:03d}.pkl', 'rb')
    fileHandleS2 = open(f'output/settings_{testNum2:03d}.pkl', 'rb')
    fileHandle1 = open(f'output/results_{testNum1:03d}_{model1}.pkl', 'rb')
    fileHandle2 = open(f'output/results_{testNum2:03d}_{model2}.pkl', 'rb')

    measure_to_optimize1, type_to_optimize1, deepROC_groups, \
    groupAxis, areaMeasures, groupMeasures      = pickle.load(fileHandleS1)
    fileHandleS1.close()
    measure_to_optimize2, type_to_optimize2, *_ = pickle.load(fileHandleS2)
    fileHandleS2.close()
    areaMatrix1, groupMatrix1                   = pickle.load(fileHandle1)
    areaMatrix2, groupMatrix2                   = pickle.load(fileHandle2)
    fileHandle1.close()
    fileHandle2.close()

    num_groups = len(deepROC_groups) + 1

    #areaMatrix   = np.zeros(shape=[total_folds, num_area_measures, iterations])
    #                               10           x5                 x100
    #groupMatrix  = np.zeros(shape=[total_folds, num_groups, num_group_measures, iterations])
    #                               10           x6          x15                 x100 = 270k * 4B = 1.08 MB

    is_not_nan = lambda a: a[np.invert(np.isnan(a))]

    vector1 = getVector(measure1, iteration1, areaMatrix1, groupMatrix1,
                        areaMeasures, groupMeasures)
    vector2 = getVector(measure2, iteration2, areaMatrix2, groupMatrix2,
                        areaMeasures, groupMeasures)
    diff    = is_not_nan(vector1 - vector2)
    mu      = np.mean(diff)
    n       = len(diff)
    if n >= 30:
        zscore = 1.96  # 5% two sided confidence
    elif n == 10:       # 9 dof
        zscore = 2.262
    elif n == 9:        # 8 dof
        zscore = 2.306
    else:
        raise ValueError('dof not anticipated')
    #endif
    confidence_interval_1sided = zscore * np.std(diff, ddof=1) / np.sqrt(n)
    lower_bound = mu - confidence_interval_1sided
    upper_bound = mu + confidence_interval_1sided
    if lower_bound < 0 and upper_bound > 0:
        print('The null hypothesis is true: '
              f'0 is inside the CI of ({lower_bound}, {upper_bound})')
    else:
        print('The null hypothesis is false: '
               f'0 is outside the CI of ({lower_bound}, {upper_bound})')
    #endif
    print(' ')
    transcript.stop()
#enddef

analyze(testNum1, model1, iteration1, measure1,
        testNum2, model2, iteration2, measure2, iterations)
