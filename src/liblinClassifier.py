__author__ = 'joel'
import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows
import cv2.cv as cv
import time
import sys
import os
import numpy
from liblinearutil import *

if(len(sys.argv) == 3):
    print '3'
    # trainingDataFile = sys.argv[1]

elif(len(sys.argv) == 4):
    print '3'
trainingDataFile = '../DatannOutput_train.csv'
trainingModelFile = '../Data/faceValidator.model'
testingDataFile = '../Data/nnOutput_test.csv'
if(os.path.isfile(trainingModelFile) == 0) :
    y, x = svm_read_problem(trainingDataFile)
    model = train(y,x,'-c 5')
    save_model(trainingModelFile,model)
else:
    model = load_model(trainingModelFile)
y,x = svm_read_problem(testingDataFile)
print 'file successfuly read in'
p_label, p_acc, p_val = predict(y, x, model)
# print len(p_val) +'examples predicted'
scoreMat = []
scoreList = 'Scores\n'
for entry in p_val:
    scoreMat.append(entry[0])
    scoreList += str(entry[0])+'\n'
outScoreFile = open('match_scores_slm.csv','a');
outScoreFile.write(scoreList)
minScore = min(scoreMat)
maxScore = max(scoreMat)
trueFalseMat = y
# outScoreFile = open('match_scores.csv','a');
# outScoreFile.write(scoreOutput)
DET = []
outString = 'Thresh,True Accept,False Accept,True Reject,False Reject\n'
print 'found all scores'
sortedScores = sorted(scoreMat)
for x in numpy.arange((minScore),(maxScore),((maxScore)-(minScore))/1000):
    trueAccept = 0.0
    falseAccept = 0.0
    falseReject = 0.0
    trueReject = 0.0
    for i in xrange(0,len(scoreMat)):
        if scoreMat[i] < x and trueFalseMat[i] == 1:
            falseReject += 1.0
        elif scoreMat[i] >= x and trueFalseMat[i] == -1:
            falseAccept += 1.0
        elif scoreMat[i] >= x and trueFalseMat[i] == 1:
            trueAccept += 1.0
        elif scoreMat[i] < x and trueFalseMat[i] == -1:
            trueReject += 1.0
    falseAccept /= (len(scoreMat)+0.0)
    falseReject /= (len(scoreMat)+0.0)
    trueAccept /= (len(scoreMat)+0.0)
    trueReject /= (len(scoreMat)+0.0)
    outString = outString + str(x) + ',' + str(trueAccept) + ',' + str(falseAccept) + ',' + str(trueReject) + ',' + str(falseReject)+'\n'
    DET.append([falseAccept,falseReject])
print DET
outFile = open('DETPointsLL.csv','a')
outFile.write(outString)
print 'file written'