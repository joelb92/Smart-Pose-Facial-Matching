# Three-layer Face Network Feature Generation Code
#
# D.D. Cox and N. Pinto, "Beyond Simple Features: A Large-Scale Feature Search
# Approach to Unconstrained Face Recognition", IEEE FG, 2011.
# 
# Implementation by Chuan-Yung Tsai (chuanyungtsai@fas.harvard.edu) 

import numpy as np
import numexpr as ne
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import normalize
import cv2.cv as cv
import time
import sys
import os

# simpler data structure
(FILT, ACTV, POOL, NORM) = range(4)
(FSIZ, FNUM, FWGH) = range(3)
(AMIN, AMAX) = range(2)
(PSIZ, PORD) = range(2)
(NSIZ, NCNT, NGAN, NTHR) = range(4)


def slminit():
    network = []

    layer = [[], [], [], []]
    layer[FILT][FSIZ:] = [0]
    layer[FILT][FNUM:] = [1]
    layer[ACTV][AMIN:] = [None]
    layer[ACTV][AMAX:] = [None]
    layer[POOL][PSIZ:] = [0]
    layer[POOL][PORD:] = [0]
    layer[NORM][NSIZ:] = [9]
    layer[NORM][NCNT:] = [0]
    layer[NORM][NGAN:] = [0.1]
    layer[NORM][NTHR:] = [1.0]
    network.append(layer)

    layer = [[], [], [], []]
    layer[FILT][FSIZ:] = [9]
    layer[FILT][FNUM:] = [128]
    layer[ACTV][AMIN:] = [0]
    layer[ACTV][AMAX:] = [None]
    layer[POOL][PSIZ:] = [9]
    layer[POOL][PORD:] = [2]
    layer[NORM][NSIZ:] = [5]
    layer[NORM][NCNT:] = [0]
    layer[NORM][NGAN:] = [0.1]
    layer[NORM][NTHR:] = [10.0]
    network.append(layer)

    layer = [[], [], [], []]
    layer[FILT][FSIZ:] = [3]
    layer[FILT][FNUM:] = [256]
    layer[ACTV][AMIN:] = [0]
    layer[ACTV][AMAX:] = [None]
    layer[POOL][PSIZ:] = [5]
    layer[POOL][PORD:] = [1]
    layer[NORM][NSIZ:] = [3]
    layer[NORM][NCNT:] = [0]
    layer[NORM][NGAN:] = [10.0]
    layer[NORM][NTHR:] = [1.0]
    network.append(layer)

    layer = [[], [], [], []]
    layer[FILT][FSIZ:] = [3]
    layer[FILT][FNUM:] = [512]
    layer[ACTV][AMIN:] = [0]
    layer[ACTV][AMAX:] = [None]
    layer[POOL][PSIZ:] = [9]
    layer[POOL][PORD:] = [10]
    layer[NORM][NSIZ:] = [5]
    layer[NORM][NCNT:] = [1]
    layer[NORM][NGAN:] = [0.1]
    layer[NORM][NTHR:] = [0.1]
    network.append(layer)

    np.random.seed(0)

    for i in xrange(len(network)):
        if (network[i][FILT][FSIZ] != 0):
            network[i][FILT][FWGH:] = [
                np.random.rand(network[i][FILT][FSIZ], network[i][FILT][FSIZ], network[i - 1][FILT][FNUM],
                               network[i][FILT][FNUM]).astype(np.float32)]
            for j in xrange(network[i][FILT][FNUM]):
                network[i][FILT][FWGH][:, :, :, j] -= np.mean(network[i][FILT][FWGH][:, :, :, j])
                network[i][FILT][FWGH][:, :, :, j] /= np.linalg.norm(network[i][FILT][FWGH][:, :, :, j])
            network[i][FILT][FWGH] = np.squeeze(network[i][FILT][FWGH])

    np.random.seed()

    return network


def nepow(X, O):
    if (O != 1):
        return ne.evaluate('X ** O')
    else:
        return X


def nediv(X, Y):
    if (np.ndim(X) == 2):
        return ne.evaluate('X / Y')
    else:
        Y = Y[:, :, None]
        return ne.evaluate('X / Y')


def nemin(X, MIN): return ne.evaluate('where(X < MIN, MIN, X)')


def nemax(X, MAX): return ne.evaluate('where(X > MAX, MAX, X)')


def mcconv3(X, W):
    X_VAW = view_as_windows(X, W.shape[0:-1])
    Y_FPS = X_VAW.shape[0:2]
    X_VAW = X_VAW.reshape(Y_FPS[0] * Y_FPS[1], -1)
    W = W.reshape(-1, W.shape[-1])
    Y = np.dot(X_VAW, W)
    Y = Y.reshape(Y_FPS[0], Y_FPS[1], -1)

    return Y


def bxfilt2(X, F_SIZ, F_STRD):
    for i in reversed(xrange(2)):
        W_SIZ = np.ones(np.ndim(X))
        S_SIZ = np.ones(2)
        W_SIZ[i], S_SIZ[i] = F_SIZ, F_STRD
        X = np.squeeze(view_as_windows(X, tuple(W_SIZ)))[::S_SIZ[0], ::S_SIZ[1]]  # subsampling before summation
        X = np.sum(X, -1)

    return X


def slmprop(X, network):
    for i in xrange(len(network)):
        if (network[i][FILT][FSIZ] != 0): X = mcconv3(X, network[i][FILT][FWGH])

        if (network[i][ACTV][AMIN] != None): X = nemin(X, network[i][ACTV][AMIN])
        if (network[i][ACTV][AMAX] != None): X = nemax(X, network[i][ACTV][AMAX])

        if (network[i][POOL][PSIZ] != 0):
            X = nepow(X, network[i][POOL][PORD])
            X = bxfilt2(X, network[i][POOL][PSIZ], 2)
            X = nepow(X, (1.0 / network[i][POOL][PORD]))

        if (network[i][NORM][NSIZ] != 0):
            B = (network[i][NORM][NSIZ] - 1) / 2
            X_SQS = bxfilt2(nepow(X, 2) if (np.ndim(X) == 2) else np.sum(nepow(X, 2), -1), network[i][NORM][NSIZ], 1)

            if (network[i][NORM][NCNT] == 1):
                X_SUM = bxfilt2(X if (np.ndim(X) == 2) else np.sum(X, -1), network[i][NORM][NSIZ], 1)
                X_MEAN = X_SUM / ((network[i][NORM][NSIZ] ** 2) * network[i][FILT][FNUM])

                X = X[B:X.shape[0] - B, B:X.shape[1] - B] - X_MEAN[:, :, None]
                X_NORM = X_SQS - ((X_SUM ** 2) / ((network[i][NORM][NSIZ] ** 2) * network[i][FILT][FNUM]))
                X_NORM = X_NORM ** (1.0 / 2)
            else:
                X = X[B:X.shape[0] - B, B:X.shape[1] - B]
                X_NORM = X_SQS ** (1.0 / 2)

            np.putmask(X_NORM, X_NORM < (network[i][NORM][NTHR] / network[i][NORM][NGAN]), (1 / network[i][NORM][NGAN]))
            X = nediv(X, X_NORM)  # numexpr for large matrix division

    return X


def genFeatres(img_list):
    network = slminit()


    filenames = img_list

    index = 0
    faceVectors = []
    for img in filenames:
        entry1 = img
        src1 = cv.LoadImageM(entry1)
        gray_full1 = cv.CreateImage(cv.GetSize(src1), 8, 1)
        grayim1 = cv.CreateImage((200, 200), 8, 1)
        cv.CvtColor(src1, gray_full1, cv.CV_BGR2GRAY)
        cv.Resize(gray_full1, grayim1, interpolation=cv.CV_INTER_CUBIC)
        gray1 = cv.GetMat(grayim1)
        im_array1 = np.asarray(gray1).astype('f')
        # -- compute feature map, shape [height, width, depth]
        f_map1 = slmprop(im_array1, network)
        f_map_dims1 = f_map1.shape
        image_vector = []
        for j in range(f_map_dims1[0]):
            for k in range(f_map_dims1[1]):
                for l in range(f_map_dims1[2]):
                    image_vector.append(f_map1[j][k][l])
        print index
        index = index+1
        faceVectors.append(np.asarray(image_vector))
    return faceVectors
    #     vector_str = str(class_label[index])
    #     for j in range(len(image_vector)):
    #         vector_index = str(j + 1)
    #         vector_str += " " + vector_index + ":" + str(image_vector[j])
    #     print index
    #     index+=1
    #     f.write(vector_str)
    #     f.write("\n")
    #
    # f.close()

if __name__ == '__main__':
    flip_faces = 0
    img_gallery_dir = sys.argv[1]
    img_probe_dir = sys.argv[2]
    posePath = os.path.join(img_probe_dir,'posetypes.txt');
    watchlist_images = [];
    probe_images = [];
    watchlist_images_fullpath = [];
    probe_images_fullpath = [];
    watchlist_labels = [];
    probe_labels = [];
    for file in os.listdir(img_gallery_dir):
        if file.endswith(".jpg"):
            watchlist_images.append(file);
            watchlist_labels.append(file[0:3])
            watchlist_images_fullpath.append(os.path.join(img_gallery_dir,file))
    for file in os.listdir(img_probe_dir):
        if file.endswith(".jpg"):
            probe_images.append(file)
            probe_images_fullpath.append(os.path.join(img_probe_dir,file))
            probe_labels.append(file[0:3])
    with open(posePath) as f:
        probePosesArray = f.readlines()
    for i in xrange(0,len(probePosesArray)):
        probePosesArray[i] = probePosesArray[i][0]
    if os.path.isfile("watchlistVectors.mat.npy"):
        watchlistVectors = np.load("watchlistVectors.mat.npy")
        print 'Generating watchlist vectors...'
    else:
        watchlistVectors = genFeatres(watchlist_images_fullpath)
        # f1 = file("watchlistVectors.mat","wb")
        np.save("watchlistVectors.mat",watchlistVectors)
        # f1.close()

    print 'Generating Probe vectors...'
    if os.path.isfile('probeVectors.mat.npy'):
        probeVectors = np.load("probeVectors.mat.npy")
    else:
        probeVectors = genFeatres(probe_images_fullpath)
        # f2 = file("probeVectors.mat","wb")
        np.save("probeVectors.mat",probeVectors)
        # f2.close()

    max_id = 0;
    for j in xrange(0,len(watchlist_images)):
        watchlist_key = watchlist_images[j]
        wLabel = int(watchlist_labels[j])
        if wLabel > max_id:
            max_id = wLabel

    similarityMatrix = np.zeros((len(probeVectors), max_id))
    for i in xrange(0,len(probeVectors)):
        probevector = probeVectors[i]
        ppose = probePosesArray[i]
        if ppose == '-1':
            ppose = '2'
        if flip_faces == 1:
            if ppose == 4:
                ppose = 3
            if ppose == 3:
                ppose = 4
        for j in xrange(0,len(watchlist_images)):
            watchlist_key = watchlist_images[j]
            wLabel = watchlist_labels[j]
            wpose = watchlist_key[6];

            if ppose == wpose:
                watchVector = watchlistVectors[j]
                dist = np.linalg.norm(probevector-watchVector)
                similarityMatrix[i,float(wLabel)-1] = dist;
    usedColumns = []
    for i in xrange(0,np.shape(similarityMatrix)[1]):
        column = similarityMatrix[:,i]
        sum = np.sum(column)
        if sum > 0:
            usedColumns.append(i)
    similarityMatrix = similarityMatrix[:, (similarityMatrix != 0).sum(axis=0) > 0]
    similarityMatrix2 = similarityMatrix / np.linalg.norm(similarityMatrix)

    mSize = np.shape(similarityMatrix)
    SortedIDMatrix = np.zeros(mSize)
    rankFound = []
    for i in xrange(0,mSize[0]):
        row = similarityMatrix[i,:]
        sortedIndexes = sorted(range(len(row)), key=lambda k: row[k])
        found = 0;
        for j in xrange(0,len(sortedIndexes)):
            ID = usedColumns[sortedIndexes[j]]+1
            if found == 0 and ID == float(probe_labels[i]):
                found = 1
                rankFound.append(j)
            SortedIDMatrix[i,j] = ID
    CMC = []
    total = 0
    for i in xrange(0,len(usedColumns)):
        total = total+rankFound.count(i)
        CMC.append(total)
    CMC = np.asarray(CMC)

    if flip_faces == 1:
        np.savetxt('SimilarityMatrix_flipped.csv',similarityMatrix, delimiter=",")
        np.savetxt('SimilarityMatrix_normalized_flipped.csv',similarityMatrix2, delimiter=",")
        np.savetxt('CMC_flipped.csv',CMC,delimiter=",")
    else:
        np.savetxt('SimilarityMatrix.csv',similarityMatrix, delimiter=",")
        np.savetxt('SimilarityMatrix_normalized.csv',similarityMatrix2, delimiter=",")
        np.savetxt('CMC.csv',CMC,delimiter=",")
