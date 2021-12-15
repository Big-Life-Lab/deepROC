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

def equal_groups(n):
    grouplist = []
    for i in range(0, n):
        grouplist = grouplist + [[i/n, (i+1)/n]]
    return grouplist
#enddef

#testNum        = 30
#reanalysis     = 'g'
testNum        = 5
reanalysis     = ''
#model         = 'case'
model          = 'plogrE2'
best_hyp       = 2   # 0-indexed
iterations     = 3
k_folds        = 10
repetition     = 1
total_folds    = k_folds * repetition
groupAxis      = 'FPR'  # FPR (=PercentileNonEvents), TPR (=PercentileEvents), Score, PercentileScore
deepROC_groups = equal_groups(6)
#deepROC_groups = [[0, 0.183], [0.183, 1]]  # FPR rf1 035ee
#deepROC_groups = [[0, 0.173], [0.173, 1]]  # FPR plogr1 035ee
#deepROC_groups = [[0, 0.178], [0.178, 1]]  # FPR xgb2 035ee
#deepROC_groups= [[0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]

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

def analyze(testNum, name, iterations, best_hyp):
    # example to load results
    if name == 'case':
        logfn = f'output/analysis-table_{name}_{testNum}.txt'
        transcript.start(logfn)
        print(f'results_{name}{testNum}:')
        fileHandle = open(f'output/results_{name}{testNum}.pkl', 'rb')
        fileHandleS = open(f'output/settings_{testNum}.pkl', 'rb')
    else:
        logfn = f'output/analysis-table_{name}_{testNum:03d}{reanalysis}.txt'
        transcript.start(logfn)
        print(f'results_{testNum:03d}{reanalysis}_{name}:')
        fileHandle = open(f'output/results_{testNum:03d}{reanalysis}_{name}.pkl', 'rb')
        fileHandleS = open(f'output/settings_{testNum:03d}.pkl', 'rb')
    #endif
    try:
        measure_to_optimize, type_to_optimize, dummy1, \
        dummy2, areaMeasures, groupMeasures = pickle.load(fileHandleS)
        fileHandleS.close()
        del dummy1, dummy2

        [areaMatrix, groupMatrix] = pickle.load(fileHandle)
        fileHandle.close()
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
    pAUC_i      = groupMeasures.index('pAUC')
    pAUCn_i     = groupMeasures.index('pAUCn')
    pAUCx_i     = groupMeasures.index('pAUCx')
    pAUCxn_i    = groupMeasures.index('pAUCxn')
    cpAUC_i     = groupMeasures.index('cpAUC')
    avgBA_i     = groupMeasures.index('avgBA')
    avgSens_i   = groupMeasures.index('avgSens')
    avgSpec_i   = groupMeasures.index('avgSpec')
    avgPPV_i    = groupMeasures.index('avgPPV')
    avgNPV_i    = groupMeasures.index('avgNPV')
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
    i = best_hyp
    #print('The following are means of overall/area measures across folds:')
    #print(f'{i:03d}: mean_AUC:        {round(100*mean_measure)/100:0.2f}')
    def print_mean_CI(measure_name, vector):
        global deepROC_groups
        is_not_nan = lambda a: a[np.invert(np.isnan(a))]
        new_vector     = is_not_nan(vector)
        n              = len(new_vector)
        if n > 0:
            mu             = np.mean(new_vector)
            mu_percent     = round(10*100*mu)/10      # percentage rounded to 86.5%
            #mu_percent     = round(1000*100*mu)/1000      # percentage rounded to 86.523%
            two_se         = 1.96 * np.std(new_vector, ddof=1) / np.sqrt(n)  # sample se uses sample std
            if not np.isnan(two_se):
                two_se_percent = round(10*100*two_se)/10  # percentage rounded to first  decimal
            else:
                two_se_percent = np.nan
            #endif
            #print(f'     {measure_name:17s} {mu_percent:>5.3f}% +/-{two_se_percent:>4.1f}%  ',
            #            f'{formatList(list(new_vector))}')
            print(f'     {measure_name:17s} {mu_percent:>5.1f}% +/-{two_se_percent:>4.1f}%  ',
                        f'{formatList(list(new_vector))}')
        else:
            print(f'     {measure_name:17s}   n/a +/- n/a  ',
                  f'{formatList(list(new_vector))}')
        #endif
    #enddef

    #for i in range(best_hyp, best_hyp+1):
    for i in range(0, iterations):
        print(f'{i}:')
        print_mean_CI('mean_AUC_full:',    areaMatrix[:,     AUC_full_i, i])
        print_mean_CI('mean_cpAUCn.all:',  groupMatrix[:, 0, cpAUC_i,    i])
        print_mean_CI(f'mean_pAUCn.all:',  groupMatrix[:, 0, pAUCn_i,    i])
        print_mean_CI(f'mean_pAUCxn.all:', groupMatrix[:, 0, pAUCxn_i,   i])
        print_mean_CI(f'mean_avgPPV.all:', groupMatrix[:, 0, avgPPV_i,   i])
        print_mean_CI(f'mean_avgNPV.all:', groupMatrix[:, 0, avgNPV_i,   i])
        print(' ')
    #endfor

    print('Within each group, the following are means across folds:')
    for g in range(1, num_groups):
        print(f'  group {g}')
        pAUC   = groupMatrix[:, g, pAUC_i,  i]
        pAUCn  = groupMatrix[:, g, pAUCn_i,  i]
        pAUCx  = groupMatrix[:, g, pAUCx_i,  i]
        pAUCxn = groupMatrix[:, g, pAUCxn_i,  i]
        # it is easy to get one of delx_v (vector) or dely_v (vector) from deepROC_groups, but not the other
        # the other varies with each fold's model, hence it is simpler if we obtain the values by reverse engineering
        # it from between the normalized and non-normalized measures...
        pAUCn_temp  = pAUCn.copy()
        pAUCxn_temp = pAUCxn.copy()
        pAUCn_temp[pAUCn_temp   == 0] = 0.01
        pAUCxn_temp[pAUCxn_temp == 0] = 0.01
        delx_v = pAUC  / pAUCn_temp
        dely_v = pAUCx / pAUCxn_temp
        print_mean_CI(f'mean_cpAUC.{g}:', groupMatrix[:, g, cpAUC_i,  i])
        print_mean_CI(f'mean_pAUC.{g}:',  pAUC)
        print_mean_CI(f'mean_pAUCx.{g}:', pAUCx)
        new_cpAUCn_i = groupMatrix[:, g, cpAUC_i,  i]
        new_cpAUCn_i = new_cpAUCn_i / (0.5 * (delx_v + dely_v))
        print_mean_CI(f'mean_cpAUCn.{g}:', new_cpAUCn_i)
        #print_mean_CI(f'mean_cpAUCn.{g}:', groupMatrix[:, g, cpAUCn_i,  i])
        print_mean_CI(f'mean_pAUCn.{g}:',  pAUCn)
        print_mean_CI(f'mean_pAUCxn.{g}:', pAUCxn)
        print(' ')
        print_mean_CI(f'mean_bAvgA.{g}:',  groupMatrix[:, g, bAvgA_i,   i])
        print_mean_CI(f'mean_avgSens.{g}:',groupMatrix[:, g, avgSens_i, i])
        print_mean_CI(f'mean_avgSpec.{g}:',groupMatrix[:, g, avgSpec_i, i])
        print(' ')
        print_mean_CI(f'mean_avgPPV.{g}:',groupMatrix[:, g, avgPPV_i, i])
        print_mean_CI(f'mean_avgNPV.{g}:',groupMatrix[:, g, avgNPV_i, i])
        print(' ')
    #endfor
    transcript.stop()
#enddef

analyze(testNum, model, iterations, best_hyp)
