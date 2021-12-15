
# import dill
import pickle
import numpy as np
import time
import transcript
import do_pAUCc  as ac2
from   sklearn   import metrics
import matplotlib.pyplot   as plt
import matplotlib.ticker   as ticker
quiet = True

def equal_groups(n):
    grouplist = []
    for i in range(0, n):
        grouplist = grouplist + [[i/n, (i+1)/n]]
    return grouplist
#enddef

analysis_name  = 'g'
testNum        = 30
model          = 'xgb2'  # model name or 'case' if a test vector
#reanalyzeOutName = f'output/reanalysis_{model}_{testNum}_{analysis_name}.pkl'
plot_hyp       = 0
hyp_iterations = 20         # iterations
k_folds        = 10
repetition     = 1
total_folds    = k_folds * repetition
posclass       = 1

  # note: whole ROC is automatically included as group 0
groupAxis      = 'TPR'  # FPR (=PercentileNonEvents), TPR (=PercentileEvents), Score, PercentileScore
deepROC_groups = equal_groups(3)
#deepROC_groups= [[0.0, 0.183], [0.183, 1]]
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

if model == 'case':  # for labelScore test vectors
    print(f'results_{model}{testNum}:')
    labelScoreFile = open(f'output/labelScore_{testNum}_{model}.pkl', 'rb')
    settingsFile   = open(f'output/settings_{testNum}.pkl', 'rb')
else:  # for labels and scores from models
    print(f'results_{testNum:03d}_{model}:')
    labelScoreFile = open(f'output/labelScore_{testNum:03d}_{model}.pkl', 'rb')
    settingsFile = open(f'output/settings_{testNum:03d}.pkl', 'rb')
# endif

tic          = time.perf_counter()

[measure_to_optimize, type_to_optimize,
    dummy1, dummy2, dummy3, dummy4] = pickle.load(settingsFile)
del dummy1, dummy2, dummy3, dummy4
settingsFile.close()

# automatically include (add) whole ROC as group 0
num_groups = len(deepROC_groups)

def plotSimpleROC(fpr,tpr,title, groupAxis, deepROC_groups):
    plt.plot(fpr, tpr)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if groupAxis == 'FPR':
        groupLines = deepROC_groups[0:len(deepROC_groups)-1][1]
        plt.vlines(groupLines, 0, 1, colors='grey', alpha=0.3)
    elif groupAxis == 'TPR':
        groupLines = deepROC_groups[0:len(deepROC_groups)-1][1]
        plt.hlines(groupLines, 0, 1, colors='grey', alpha=0.3)
    #endif
    plt.show()
#enddef

for hyp in range(0, plot_hyp+1):
#for hyp in range(0, 16):
    try:
        # labels, scores - a 2D matrix for each hyp of [fold][instance]
        [labels, scores] = pickle.load(labelScoreFile)
    except:
        print('pickle load failed')
        exit(1)
    #endtry
    if hyp != plot_hyp:
        continue
    #endif
    # print(f'iteration {hyp}')

    # compute measures for each fold
    #for fold in range(0, 1):  # show only the first fold
    for fold in range(0, total_folds):  # show all folds
        P = sum(labels[fold] == 1)  # number of actual positives (assume higher value)
        N = sum(labels[fold] == 0)  # number of actual negatives (assume lower value)
        globalP = P
        globalN = N
        logTestNum = f'{testNum}-{model}-{fold}'

        scores[fold], newlabels, labels[fold] = ac2.sortScoresFixLabels(scores[fold], labels[fold], posclass, True)
        # True = ascending

        ffpr, ftpr, fthresh, fnewlabel, fSlopeFactor = ac2.getFullROC(list(labels[fold]), list(scores[fold]), posclass)

        ffpr         = np.delete(ffpr, 0)
        ftpr         = np.delete(ftpr, 0)
        fthresh      = np.delete(fthresh, 0)
        fnewlabel    = np.delete(fnewlabel, 0)
        fSlopeFactor = np.delete(fSlopeFactor, 0)

        posHistogram = np.zeros([num_groups])
        negHistogram = np.zeros([num_groups])
        for g in range(0, num_groups):
            pos = np.array(labels[fold]) == posclass
            neg = np.array(labels[fold]) != posclass
            if groupAxis == 'FPR':
                # index of first FPR value within the group, where ffpr is in ascending order
                indexFirstInGroup = np.argwhere(ffpr >= deepROC_groups[g][0])[0]
                indexLastInGroup  = np.argwhere(ffpr <= deepROC_groups[g][1])[-1]
            elif groupAxis == 'TPR':
                indexFirstInGroup = np.argwhere(ftpr >= deepROC_groups[g][0])[0]
                indexLastInGroup  = np.argwhere(ftpr <= deepROC_groups[g][1])[-1]
            # endif
            # now because thresh is descending order, inside start is LESS THAN, inside end is GREATER THAN
            withinStart = np.array(scores[fold]) <= float(fthresh[indexFirstInGroup])
            withinEnd   = np.array(scores[fold]) >= float(fthresh[indexLastInGroup ])
            posHistogram[g] = sum(pos & withinStart & withinEnd)
            negHistogram[g] = sum(neg & withinStart & withinEnd)
            #posHistogram[num_groups - g - 1] = sum(pos & withinStart & withinEnd)
            #negHistogram[num_groups - g - 1] = sum(neg & withinStart & withinEnd)
        #endfor

        total = sum(posHistogram) + sum(negHistogram)
        fpr, tpr, thresholds = metrics.roc_curve(labels[fold], scores[fold], pos_label=posclass)
        title = f'ROC plot for fold {fold}'
        plotSimpleROC(fpr, tpr, title, groupAxis, deepROC_groups)

        fig = plt.figure(figsize=(10, 5))
        # creating the bar plot
        v = [1, 2, 3]
        #v = [1, 2, 3, 4, 5, 6]
        #v = [1, 2]

        w = posHistogram.copy()
        x = negHistogram.copy()
        y = posHistogram.copy()
        z = negHistogram.copy()
        w[0:2] = 0   # pos foreground zeroized
        y[2:3] = 0   # pos background zeroized
        x[2:3] = 0   # neg foreground zeroized
        z[0:2] = 0   # neg background zeroized

        #w[1:2] = 0   # pos foreground zeroized
        #y[0:1] = 0   # pos background zeroized
        #x[0:1] = 0
        #z[1:2] = 0

        #w[3:6] = 0
        #y[0:3] = 0
        #x[0:3] = 0
        #z[3:6] = 0

        #w[5:6] = 0  # pos foreground zeroized
        #y[0:4] = 0  # pos background zeroized
        #x[0:5] = 0  # neg foreground zeroized
        #z[5:6] = 0  # neg background zeroized

        # plot higher background color
        plt.bar(v, y, color='blue', alpha=0.8, width=0.4)  # positives background
        plt.bar(v, z, color='red', alpha=1.0, width=0.4)

        plt.bar(v, w, color='blue', alpha=0.8, width=0.4)  # positives foreground
        plt.bar(v, x, color='red', alpha=1.0, width=0.4)   # neg foreground
        plt.ylabel("Adults")
        #plt.title("")
        plt.show()

        #if index == 0 or index == 1:  # index 1,2,... for partial curves, 0 for whole
        #    rocRuleLeft = 'SW'
        #    rocRuleRight = 'NE'
        #else:
        #    rocRuleLeft = 'NE'
        #    rocRuleRight = 'NE'
        ##endif
        #doMyPlot(y_cv_scores[fold], y_cv_s[fold], groupAxis, deepROC_groups, logTestNum)

    #endfor
#endfor
labelScoreFile.close()