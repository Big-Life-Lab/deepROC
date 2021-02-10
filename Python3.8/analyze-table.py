# analyze-table.py
# acLogging.py
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
#
#   Note: this file has not been updated to match others like analyze15.py

import dill
import pickle
import numpy as np
import transcript

testNum        = 5
#model         = 'case'
model          = 'rf0'
best_iteration = 37  # 0-indexed
iterations     = 200
k_folds        = 5
repetition     = 1
total_folds    = k_folds * repetition

def formatList(alist):
    # Create a format spec for each item in the input `alist`.
    # E.g., each item will be right-adjusted, field width=3.
    format_list = ['{:0.5f}' for item in alist]

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

def analyze(testNum, name, iterations, best_iteration):
    # example to load results
    if name == 'case':
        logfn = f'output/analysis-table_{name}{testNum}.txt'
        transcript.start(logfn)
        print(f'results_{name}{testNum}:')
        fileHandle = open(f'output/results_{name}{testNum}.pkl', 'rb')
    else:
        logfn = f'output/analysis-table_{name}{testNum:03d}.txt'
        transcript.start(logfn)
        print(f'results_{testNum:03d}_{name}:')
        fileHandle = open(f'output/results_{testNum:03d}_{name}.pkl', 'rb')
    #endif
    try:
        measure_to_optimize, type_to_optimize, deepROC_groups, \
        groupAxis, areaMeasures, groupMeasures = pickle.load(fileHandle)
    except:
        print('pickle load failed')
        exit(1)
    #endtry

    # create indices from above
    AUC_i       = areaMeasures.index('AUC')
    AUC_full_i  = areaMeasures.index('AUC_full')
    AUC_plain_i = areaMeasures.index('AUC_plain')
    AUC_micro_i = areaMeasures.index('AUC_micro')
    AUPRC_i     = areaMeasures.index('AUPRC')
    cpAUCn_i    = groupMeasures.index('cpAUCn')
    pAUCn_i     = groupMeasures.index('pAUCn')
    pAUCxn_i    = groupMeasures.index('pAUCxn')
    cpAUC_i     = groupMeasures.index('cpAUC')
    avgBA_i     = groupMeasures.index('avgBA')
    avgSens_i   = groupMeasures.index('avgSens')
    avgSpec_i   = groupMeasures.index('avgSpec')
    bAvgA_i     = groupMeasures.index('bAvgA')

    num_groups         = len(deepROC_groups) + 1
    num_group_measures = len(groupMeasures)
    num_area_measures  = len(areaMeasures)

    #mean_measure = [None] * total_folds
    #areaMatrix   = np.zeros(shape=[total_folds, num_area_measures, iterations])
    #                               5            x5                 x100
    #groupMatrix  = np.zeros(shape=[total_folds, num_groups, num_group_measures, iterations])
    #                               5            x6          x15                 x100 = 270k * 4B = 1.08 MB

    # code from https://stackoverflow.com/questions/7568627/using-python-string-formatting-with-lists
    #for i in range(0, iterations+1):
    for i in range(0, best_iteration+1):
        #try:
        [mean_measure, areaMatrix, groupMatrix] = pickle.load(fileHandle)
        if i < best_iteration:
            continue
        print('The following are means of overall/area measures across folds:')
        print(f'{i:03d}: mean_AUC:        {round(100*mean_measure)/100:0.2f}')
        def print_mean_CI(measure_name, vector):
            n              = len(vector)
            mu             = np.mean(vector)
            mu_percent     = round(10*100*mu)/10      # percentage rounded to second decimal
            two_se         = 1.96 * np.std(vector, ddof=1) / np.sqrt(n)  # sample se uses sample std
            two_se_percent = round(10*100*two_se)/10  # percentage rounded to first  decimal
            print(f'     {measure_name:17s} {mu_percent:>5.1f}% +/-{two_se_percent:>4.1f}%  ',
                  f'{formatList(list(vector))}')
        #enddef

        print_mean_CI('mean_AUC_full:', areaMatrix[:,    AUC_full_i, i])
        print_mean_CI('mean_cpAUC.0:', groupMatrix[:, 0, cpAUC_i,    i])
        print(' ')

        print('Within each group, the following are means across folds:')
        for g in range(1, num_groups):
            print(f'  group {g}')
            print_mean_CI(f'mean_cpAUCn.{g}:', groupMatrix[:, g, cpAUCn_i,  i])
            print_mean_CI(f'mean_pAUCn.{g}:',  groupMatrix[:, g, pAUCn_i,   i])
            print_mean_CI(f'mean_pAUCxn.{g}:', groupMatrix[:, g, pAUCxn_i,  i])
            print(' ')
            print_mean_CI(f'mean_bAvgA.{g}:',  groupMatrix[:, g, bAvgA_i,   i])
            print_mean_CI(f'mean_avgSens.{g}:',groupMatrix[:, g, avgSens_i, i])
            print_mean_CI(f'mean_avgSpec.{g}:',groupMatrix[:, g, avgSpec_i, i])
            print(' ')
            #endfor
        #except:
            #break
        #endtry
    #endfor
    fileHandle.close()
    transcript.stop()
#enddef

analyze(testNum, model, iterations, best_iteration)
