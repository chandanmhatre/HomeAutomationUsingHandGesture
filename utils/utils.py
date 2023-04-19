import os
import pickle

import numpy as np

mainModelPath = 'dataset/MainModel/main.pkl'
newGestureSavingPath = 'dataset/model/'
gesturePlugPinCsvPath = 'dataset/gesture-plug-pin.csv'
detectionKeypointItteration = 10


class Utils(object):
    def buildMainModel(self):
        newGestNames = []
        newKnownGestures = []
        for item in os.listdir(newGestureSavingPath):
            if item.endswith('.pkl'):
                with open(newGestureSavingPath+item, 'rb') as f:
                    gestNames2 = pickle.load(f)
                    knownGestures2 = pickle.load(f)
                    for data in gestNames2:
                        newGestNames.append(data)
                    for data in knownGestures2:
                        newKnownGestures.append(data)
        with open(mainModelPath, 'wb') as out:
            pickle.dump(newGestNames, out, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(newKnownGestures, out,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def findDistances(self, handData):
        distMatrix = np.zeros([len(handData), len(handData)], dtype='float')
        palmSize = ((handData[0][0]-handData[9][0])**2 +
                    (handData[0][1]-handData[9][1])**2)**(1./2.)
        for row in range(0, len(handData)):
            for column in range(0, len(handData)):
                distMatrix[row][column] = (((handData[row][0]-handData[column][0])**2+(
                    handData[row][1]-handData[column][1])**2)**(1./2.))/palmSize
        return distMatrix

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def findGesture(self, unknownGesture, knownGestures, keyPoints, gestNames, tol):
        errorArray = []
        for i in range(0, len(gestNames) * detectionKeypointItteration, 1):
            error = Utils.findError(
                knownGestures[i], unknownGesture, keyPoints)
            errorArray.append(error)
        errorMin = errorArray[0]
        for i in range(0, len(errorArray), 1):
            if errorArray[i] < errorMin:
                errorMin = errorArray[i]  # lowest value is correct gesture
        minIndex = 0
        errorSplitList = list(Utils.split(errorArray, len(gestNames)))
        for i in range(0, len(errorSplitList), 1):
            if errorMin in errorSplitList[i]:
                minIndex = i
        if errorMin < tol:  # tol = 10 , if error min < 10 is valid gesture
            gesture = gestNames[minIndex]
            # print('++++++++++++++++++++++++++++')
            # print(errorArray)
            # print(gesture)
        if errorMin >= tol:
            gesture = 'Unknown'
        return gesture

    def findError(gestureMatrix, unknownMatrix, keyPoints):
        error = 0
        for row in keyPoints:
            for column in keyPoints:
                error = error+abs(gestureMatrix[row]
                                  [column]-unknownMatrix[row][column])
        return error
