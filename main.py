import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from skimage import data, color, exposure
from image import Image
from operator import itemgetter
import time
import math

def kaggle(testNumber, result, f):
	string = str(testNumber) + "," + str(result) + "\n"
	f.append(string)
	return f

def printFile(content, fileName):
	open(fileName, "w").close()
	fileOut = open(fileName, "a")
	for line in content:
		fileOut.write(line)
	fileOut.close()

def getAttributeDict():
	fileTrain = open("train.txt", "r").read().splitlines()
	attributeDict={}
	for im_loc in fileTrain:
		attributeDict = getNameID(im_loc, attributeDict)
	return attributeDict

def getAttributeDictFolded():
	fileTrain = open("train.txt", "r").read().splitlines()
	trainDict={}
	testDict={}
	for i, im_loc in enumerate(fileTrain):
		if (i>(len(fileTrain)/5)):
			trainDict = getNameID(im_loc, trainDict)
		else:
			testDict = getNameID(im_loc, testDict)
	return trainDict, testDict

def getTestAttribute():
	attributeDict={}
	for i in range(1000):
		attributeDict[str(i+1) + ".jpg"] = i
	return attributeDict

def getNameID(loc, nameId):
	ID = int(loc.split(" ")[0])
	name = loc.split(" ")[1]
	nameId[name] = ID
	return nameId

def train(locationIm, locationAn, attributeAndName, attributes, dataSet, test=0):
	inst = Image(locationIm, locationAn, attributeAndName, attributes, test)
	dataSet.append(inst)
	return dataSet

def calcDistanceL1(test, dataSet):
	l1Distances = []		#((x,y),bclass)
	for data in dataSet:
		l1Distances.append(((np.sum(np.abs(np.subtract(data.colorHistogram, test.colorHistogram))), np.sum(np.abs(np.subtract(data.attribute, test.attribute)))),data.bclass))
	return l1Distances

def calcDistanceL2(test, dataSet):
	l2Distances = []
	for data in dataSet:
		l2Distances.append(((math.sqrt(np.sum(np.abs(np.subtract(data.colorHistogram, test.colorHistogram)**2))), math.sqrt(np.sum(np.abs(np.subtract(data.attribute, test.attribute)**2)))),data.bclass))
	return l2Distances

def knnCalculate(distance, histW, attW, k):
	distances = list(map(lambda feature : [math.sqrt((feature[0][0]*histW)**2+(feature[0][1]*attW)**2), feature[1]], distance))
	#minimum = min(distances, key=itemgetter(0))
	sortedDist = sorted(distances, key=itemgetter(0))[:k]
	freqDict = {}
	for elm in sortedDist:
		try:
			freqDict[elm[1]] = freqDict[elm[1]] + 1
		except:
			freqDict[elm[1]] = 1
	moreThanOneResult = 0
	maxim = 0
	res = -1
	dummyList = []
	for k, v in freqDict.items():
		if v > maxim:
			maxim = v
			moreThanOneResult = 0
			res = k
			dummyList = [k]
		elif v == maxim:
			moreThanOneResult = 1
			dummyList.append(k)
	if (moreThanOneResult == 0):
		return res
	else:
		for elm in sortedDist:
			if elm[1] in dummyList:
				return elm[1]

def main(histogramWeight, attributeWeight, distType, k):
	attributeAndName = getAttributeDict()
	attributes = np.load('attributes.npy')
	attributes_test = np.load('test_attributes.npy')
	dataSet = []


	startTime = time.time()
	for elm in attributeAndName:
		locationIm = './images/images/'+ elm
		locationAn = './annotations/annotations-mat/' + ".".join(elm.split(".")[:-1]) + ".mat"
		dataset = train(locationIm, locationAn, attributeAndName, attributes, dataSet)
	endTime = time.time()
	print ("Trained training group in {} seconds".format(endTime-startTime))

	testSet = []
	testAttributeAndName = getTestAttribute()
	startTime = time.time()
	for elm in testAttributeAndName:
		locationIm = './test_images/' + elm
		locationAn = './test_annotations/' + ".".join(elm.split(".")[:-1]) + ".mat"
		testSet = train(locationIm, locationAn, testAttributeAndName, attributes_test, testSet,1)
	endTime = time.time()
	print ("Trained test group in {} seconds".format(endTime-startTime))

	f = ["Id,Category\n"]
	start = time.time()
	for i in range(len(testSet)):
		test = testSet[i]
		if (distType == "l1"):
			distances = calcDistanceL1(test, dataSet) #(test instance, set of trained images)
		elif (distType == "l2"):
			distances = calcDistanceL2(test, dataSet)
		result = knnCalculate(distances,histogramWeight,attributeWeight, k)#, weight of color histogram, weight of attributes)
		#print(result)
		f = kaggle(int(test.name.split(".")[0]), int(result), f)
		#print(test.name, result)
	print("Calculated distance in {} seconds".format(time.time()-start))
	printFile(f, "kaggle.csv")

	'''
	THE PART BELOW IS FOR TESTING WITH A PART OF TRAINING DATA


	trainDict, testDict = getAttributeDictFolded()
	startTime = time.time()
	for elm in trainDict:
		locationIm = './images/images/'+ elm
		locationAn = './annotations/annotations-mat/' + ".".join(elm.split(".")[:-1]) + ".mat"
		dataset = train(locationIm, locationAn, attributeAndName, attributes, dataSet)
	endTime = time.time()
	#print ("Trained training group in {} seconds".format(endTime-startTime))

	testSet = []
	startTime = time.time()
	for elm in testDict:
		locationIm = './images/images/'+ elm
		locationAn = './annotations/annotations-mat/' + ".".join(elm.split(".")[:-1]) + ".mat"
		testSet = train(locationIm, locationAn, attributeAndName, attributes, testSet)
	endTime = time.time()
	#print ("Trained test group in {} seconds".format(endTime-startTime))

	start = time.time()
	correct = 0
	false = 0
	for i in range(len(testSet)):
		test = testSet[i]
		if (distType == "l1"):
			distances = calcDistanceL1(test, dataSet) #(test instance, set of trained images)
		elif (distType == "l2"):
			distances = calcDistanceL2(test, dataSet)
		result = knnCalculate(distances,histogramWeight,attributeWeight, k)#, weight of color histogram, weight of attributes, k)
		if (test.bclass == result):
			correct = correct + 1
		else:
			false = false + 1
		#print(result)
	print("Calculated distance in {} seconds".format(time.time()-start))
	print("Accuracy for {} distance {}nn is {}% for histogram weight {}, and attribute weight {}.".format(distType, k, (correct/(correct+false))*100, histogramWeight, attributeWeight))



	THAT PART ENDS HERE
	'''


distType = "l2"
attributeWeight = 0.5
histogramWeight = 1e-4
k = 5
main(histogramWeight, attributeWeight, distType, k)





#
