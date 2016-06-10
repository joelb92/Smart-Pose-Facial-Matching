__author__ = 'joel'
from brpy import init_brpy
import brpy
import cv2
import numpy
import sys
import getopt
import os
import subprocess

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

def main(argv):
    br = init_brpy()
    br.br_initialize_default()
    br.br_set_property('algorithm','FaceRecognition') # also made up
    # br.br_set_property('enrollAll','true')
    inputfile = ''
    outputfile = ''
    dataLocation = ''
    try:
        opts, args = getopt.getopt(argv,"hi:d",["ifile=","dfile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-d", "--dfile"):
            dataLocation = '~/openbr/data/LFW/img/'
    lines = [line.rstrip('\n') for line in open(inputfile)]
    lines = lines[1:]
    scoreMat = []
    trueFalseMat = []
    numTrue = 0
    numFalse = 0
    i = 0
    scoreOutput = 'file1,file2,score,true\n'
    for line in lines :
        components = line.split("\t");
        im1Path = ''
        im2Path = ''
        isTrue = 0
        if len(components) == 3 :
            trueFalseMat.append(1)
            isTrue = 1
            numTrue +=1
            folder = components[0]
            im1num = components[1].zfill(4)
            im2num = components[2].zfill(4)
            im1Path = dataLocation + folder + '/' + folder + '_' + im1num + '.jpg'
            im2Path = dataLocation + folder + '/' + folder + '_' + im2num + '.jpg'
        elif len(components) == 4:
            trueFalseMat.append(0)
            isTrue = 0
            numFalse += 1
            folder1 = components[0]
            folder2 = components[2]
            im1num = components[1].zfill(4)
            im2num = components[3].zfill(4)
            im1Path = dataLocation + folder1 + '/' + folder1 + '_' + im1num + '.jpg'
            im2Path = dataLocation + folder2 + '/' + folder2 + '_' + im2num + '.jpg'
        im1 = open(im1Path, 'rb').read()
        im2 = open(im2Path, 'rb').read()
        p1 = subprocess.Popen(["br","-algorithm", "FaceRecognition" ,"-compare", im1Path, im2Path], stdout=subprocess.PIPE)
        score = float(p1.communicate()[0])
        if score < -100:
            score = -10
        elif score > 100:
            score = 10
        p1.stdout.close()
        scoreOutput = scoreOutput+im1Path+','+im2Path+','+ str(score) + str(isTrue) + '\n'
        # templ1 = br.br_load_img(im1, len(im1))
        # templ2 = br.br_load_img(im2, len(im2))
        # query1 = br.br_enroll_template(templ1)
        # query2 = br.br_enroll_template(templ2)
        # score = br.br_compare_template_lists(query1,query2)
        # br.br_pairwise_compare(query1,query2,1)
        scoreMat.append(score)
        print i
        i+=1
    medianScore = numpy.median(scoreMat)

    minScore = min(scoreMat)
    maxScore = max(scoreMat)
    outScoreFile = open('match_scores.csv','a');
    outScoreFile.write(scoreOutput)
    DET = []
    outString = 'Thresh, True Accept,False Accept,True Reject,False Reject\n'
    print 'found all scores'
    sortedScores = sorted(scoreMat)
    arr = numpy.arange((minScore),(maxScore),((maxScore)-(minScore))/1000)
    for x in numpy.arange((minScore),(maxScore),((maxScore)-(minScore))/1000):
        trueAccept = 0.0
        falseAccept = 0.0
        falseReject = 0.0
        trueReject = 0.0
        for i in xrange(0,len(scoreMat)):
            if scoreMat[i] < x and trueFalseMat[i] == 1:
                falseReject += 1.0
            elif scoreMat[i] >= x and trueFalseMat[i] == 0:
                falseAccept += 1.0
            elif scoreMat[i] >= x and trueFalseMat[i] == 1:
                trueAccept += 1.0
            elif scoreMat[i] < x and trueFalseMat[i] == 0:
                trueReject += 1.0
        falseAccept /= (len(scoreMat)+0.0)
        falseReject /= (len(scoreMat)+0.0)
        trueAccept /= (len(scoreMat)+0.0)
        trueReject /= (len(scoreMat)+0.0)
        outString = outString + str(x)+ ','+ str(trueAccept) + ',' + str(falseAccept) + ',' + str(trueReject) + ',' + str(falseReject)+'\n'
        DET.append([falseAccept,falseReject])
    print DET
    outFile = open('DETPoints.csv','a');
    outFile.write(outString)


if __name__ == "__main__":
   main(sys.argv[1:])
