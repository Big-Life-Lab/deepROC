
# import dill
import pickle
import numpy as np
import time
import transcript
import deepROC   as ac

quiet = True

def equal_groups(n):
    grouplist = []
    for i in range(0, n):
        grouplist = grouplist + [[i/n, (i+1)/n]]
    return grouplist
#enddef

analysis_name  = 'a'
testNum        = 27
model          = 'xgb2'  # model name or 'case' if a test vector
#reanalyzeOutName = f'output/reanalysis_{model}_{testNum}_{analysis_name}.pkl'
hyp_iterations = 10         # iterations
k_folds        = 10
repetition     = 1
total_folds    = k_folds * repetition

  # note: whole ROC is automatically included as group 0
groupAxis      = 'FPR'  # FPR (=PercentileNonEvents), TPR (=PercentileEvents), Score, PercentileScore
deepROC_groups = equal_groups(3)
#deepROC_groups= [[0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]
#deepROC_groups= [[0.0, 0.167], [0.167, 0.333], [0.333, 0.5],
#                  [0.5, 0.667], [0.667, 0.833], [0.833, 1.0]]

reportBestFor = ['area','AUC']
reportMeasures = [['area' , 'AUC']   , ['area', 'AUPRC']  ,
                  ['group', 'cpAUCn']  ,
                  ['group', 'pAUCn'] , ['group', 'pAUCxn'],
                  ['group', 'avgPPV'], ['group', 'avgNPV']]
areaMeasures   = ['AUC'   , 'AUC_full', 'AUC_plain', 'AUC_micro', 'AUPRC']
groupMeasures  = ['cDelta', 'cpAUC'   , 'pAUC'     , 'pAUCx',
                  'cDeltan','cpAUCn'  , 'pAUCn'    , 'pAUCxn',
                  'avgA'  , 'bAvgA'   , 'avgSens'  , 'avgSpec',
                  'avgPPV', 'avgNPV'  , 'avgLRp'   , 'avgLRn',
                  'ubAvgA', 'avgBA'   , 'sPA']
num_group_measures = len(groupMeasures)
num_area_measures  = len(areaMeasures)

##########
# Setup advanced performance measures
def doAdvancedMeasures(scores, labels, groupAxis, deepROC_groups, testNum):
    global quiet, globalP, globalN
    costs = dict(cFP=1, cFN=1, cTP=0, cTN=0, rates=False)
    results, EQresults = ac.deepROC(costs=costs,     showPlot=False,  showData=False, showError=False,
                                    globalP=globalP, globalN=globalN, scores=scores,   labels=labels,  posclass=1,
                                    testNum=testNum, pAUCranges=deepROC_groups,   rangeAxis=groupAxis,
                                    useCloseRangePoint=False, sanityCheckWholeAUC=True, quiet=quiet)
    return results, EQresults
#enddef

def convertMeasureToIndices(type_to_index, measure_to_index, areaMeasures, groupMeasures):
    #######
    # Convert measure_to_optimize into indices
    measure_index = False
    group_index   = False
    i = 0
    if type_to_index == 'area':
        for m in areaMeasures:
            if measure_to_index == m:
                measure_index = i
                group_index = False
            # endif
            i = i + 1
        # endfor
    elif type_to_index == 'group':
        name_to_index , num_string = measure_to_index.split('.')
        for m in groupMeasures:
            if name_to_index == m:
                measure_index = i
                group_index = int(num_string)
            # endif
            i = i + 1
        # endfor
    # endif
    return measure_index, group_index
#enddef

if model == 'case':  # for labelScore test vectors
    logfn          = f'output/reanalysis_{model}_{testNum}_{analysis_name}.txt'
    transcript.start(logfn)
    print(f'results_{model}{testNum}:')
    labelScoreFile = open(f'output/labelScore_{testNum}_{model}.pkl', 'rb')
    settingsFile   = open(f'output/settings_{testNum}.pkl', 'rb')
else:  # for labels and scores from models
    logfn          = f'output/reanalysis_{model}_{testNum:03d}_{analysis_name}.txt'
    transcript.start(logfn)
    print(f'results_{testNum:03d}_{model}:')
    labelScoreFile = open(f'output/labelScore_{testNum:03d}_{model}.pkl', 'rb')
    settingsFile = open(f'output/settings_{testNum:03d}.pkl', 'rb')
# endif

tic          = time.perf_counter()

[measure_to_optimize, type_to_optimize,
    dummy1, dummy2, dummy3, dummy4] = pickle.load(settingsFile)
del dummy1, dummy2, dummy3, dummy4
settingsFile.close()

print(f'During training the data were optimized for {type_to_optimize} measure: {measure_to_optimize}')
print(f'We are reporting the best: {reportBestFor[1]} (an {reportBestFor[0]} measure)')

# automatically include (add) whole ROC as group 0
num_groups = len(deepROC_groups) + 1
areaMatrix = np.zeros(shape=[total_folds, num_area_measures, hyp_iterations])
#                              30           x5                 x100
groupMatrix = np.zeros(shape=[total_folds, num_groups, num_group_measures, hyp_iterations])
#                              30           x6          x15                 x100 = 270k * 4B = 1.08 MB
y_cv_s      = np.zeros(shape=[hyp_iterations])
y_cv_scores = np.zeros(shape=[hyp_iterations])

# the following lambda function allows us to handle np.nan when computing mean, sum, etc with results
is_not_nan = lambda a: a[np.invert(np.isnan(a))]

for hyp in range(0, hyp_iterations):
#for hyp in range(0, 16):
    try:
        # labels, scores - a 2D matrix for each hyp of [fold][instance]
        [y_cv_s, y_cv_scores] = pickle.load(labelScoreFile)
    except:
        print('pickle load failed')
        exit(1)
    #endtry

    # print(f'iteration {hyp}')

    # compute measures for each fold
    for fold in range(0, total_folds):
        P = sum(y_cv_s[fold] == 1)  # number of actual positives (assume higher value)
        N = sum(y_cv_s[fold] == 0)  # number of actual negatives (assume lower value)
        globalP = P
        globalN = N
        logTestNum = f'{testNum}-{model}-{fold}'
        results, EQresults = doAdvancedMeasures(y_cv_scores[fold], y_cv_s[fold],
                                                groupAxis, deepROC_groups, logTestNum)

        for group in range(0, num_groups):
            # note: these are validation results, not training results
            if group == 0:
                areaMatrix[fold, :, hyp]     = np.array([results[group][m] for m in areaMeasures])
            #endif
            groupMatrix[fold, group, :, hyp] = np.array([results[group][m] for m in groupMeasures])
        #endfor
    #endfor
#endfor
labelScoreFile.close()

def getShowBestResult(reportBestFor, areaMeasures, groupMeasures, areaMatrix, groupMatrix):
    global hyp_iterations, is_not_nan
    vector_of_means = np.zeros(shape=[hyp_iterations])
    #
    # report measures
    [type_to_index, measure_to_index] = reportBestFor
    #
    if type_to_index == 'area':
        bestForMeasure_index, bestForGroup_index = \
            convertMeasureToIndices(type_to_index, measure_to_index, areaMeasures, groupMeasures)
        for t in range(0, hyp_iterations):
            vector_of_means[t] = np.mean(is_not_nan( areaMatrix[:, bestForMeasure_index, t] ), axis=0) # mean across folds
        #endfor
        #
        bestMean         = np.max(vector_of_means)  # max across trials
        dummy1           = np.argwhere(vector_of_means == bestMean)
        bestHypIteration = int(np.argwhere(vector_of_means == bestMean)[0])
        print(f'The best mean {measure_to_index} is {bestMean:0.4f} in trial #{bestHypIteration:d} (0-indexed)')
        #
    elif type_to_index == 'group':
        measure_and_group = f'{measure_to_index}.{group:d}'
        bestForMeasure_index, bestForGroup_index = \
            convertMeasureToIndices(type_to_index, measure_and_group, areaMeasures, groupMeasures)
        for t in range(0, hyp_iterations):
            vector_of_means[t] = np.mean(is_not_nan(groupMatrix[:, bestForGroup_index, bestForMeasure_index, t]), axis=0) 
            # mean across folds
        #endfor
        #
        bestMean         = np.max(vector_of_means) # max across trials
        bestHypIteration = int(np.argwhere(vector_of_means == bestMean))
        print(f'The best mean {measure_to_index} is {bestMean:0.4f} in trial {bestHypIteration:d} (0-indexed)')
    #endif
    return bestMean, bestHypIteration, vector_of_means # no nan/missing in vector_of_means expected
#enddef

def showAllResultsForHyp(reportMeasures, hyp_iteration, areaMeasures, groupMeasures, areaMatrix, groupMatrix):
    global is_not_nan
    reportMeasureValues = np.zeros(shape=[len(reportMeasures)])
    # report measures
    i = 0
    for m in reportMeasures:
        [type_to_index, measure_to_index] = m
        if   type_to_index == 'area':
            measure_index, group_index = convertMeasureToIndices(type_to_index, measure_to_index, areaMeasures, groupMeasures)
            reportMeasureValues[i] = np.mean(is_not_nan(areaMatrix[:, measure_index, hyp_iteration]), axis=0) # mean across folds
            print(f'{measure_to_index:11s}: {reportMeasureValues[i]:0.4f}')
        #
        elif type_to_index == 'group':
            for group in range(0, num_groups):
                measure_and_group=f'{measure_to_index}.{group:d}' 
                measure_index, group_index = convertMeasureToIndices(type_to_index, measure_and_group, areaMeasures, groupMeasures)
                reportMeasureValues[i] = np.mean(is_not_nan(groupMatrix[:, group_index, measure_index, hyp_iteration]), axis=0)
                # mean across folds
                print(f'{measure_and_group:11s}: {reportMeasureValues[i]:0.4f}')
            #endfor
        #endif
    #endfor
    return reportMeasureValues
#enddef

bestMean, bestHypIteration, vector_of_means = \
    getShowBestResult(reportBestFor, areaMeasures, groupMeasures, areaMatrix, groupMatrix)

reportMeasureValues = \
    showAllResultsForHyp(reportMeasures, bestHypIteration, areaMeasures, groupMeasures, areaMatrix, groupMatrix)

print(' ')
transcript.stop()

# save results and settings for analyze15.py
# results: [areaMatrix, groupMatrix]
# settings: [measure_to_optimize, type_to_optimize, deepROC_groups, groupAxis, areaMeasures, groupMeasures]
resultFileOut = open(f'output/results_{testNum:03d}{analysis_name}_{model}.pkl', 'wb')
pickle.dump([areaMatrix, groupMatrix], resultFileOut, pickle.HIGHEST_PROTOCOL)

#settingsFileOut = open(f'output/settings_{testNum:03d}{analysis_name}.pkl', 'wb')
#pickle.dump([measure_to_optimize, type_to_optimize, deepROC_groups, groupAxis, areaMeasures, groupMeasures],
#            settingsFileOut, pickle.HIGHEST_PROTOCOL)

#pickle.dump([reportBestFor, areaMeasures, groupMeasures, areaMatrix, groupMatrix,
#             bestMean, bestHypIteration, vector_of_means, reportMeasures, reportMeasureValues],
#             reanalyzeOutFile, pickle.HIGHEST_PROTOCOL)
