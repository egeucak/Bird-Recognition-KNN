import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog
from skimage import data, color, exposure

class Image:
	def __init__(self, locationIm, locationAn, attributeAndName, attributes, test=0):
		try:
			self.location = locationIm
			self.annotationLoc = locationAn
			self.image = self.__imread(self.location)
			if (test == 0):
				self.bclass = self.location.split("/")[3].split(".")[0]
				self.name = "/".join(locationIm.split("/")[3:5])
			else:
				self.bclass = ""
				self.name = locationIm.split("/")[-1]
			self.colorHistogram = self.__ColorHistogram()
			self.attribute = self.__getAttribute(attributeAndName, attributes)
			print
		except:
			i = 0
			print("Missing file")

	def __ColorHistogram(self, numofbin = 256):
		rgbarray = self.image
		segment = sio.loadmat(self.annotationLoc)['seg']
		histB = cv2.calcHist([rgbarray],[0],segment,[256],[0,256])
		histG = cv2.calcHist([rgbarray],[1],segment,[256],[0,256])
		histR = cv2.calcHist([rgbarray],[2],segment,[256],[0,256])
		hist = np.concatenate((histR, histG, histB), axis=0)
		return hist

	def __imread(self, loc):
		img=mpimg.imread(loc)
		return img

	def __getAttribute(self, attributeAndName, attributes):
		return attributes[attributeAndName[self.name]]

	def __boundImage(self):
		bounding_box = sio.loadmat(self.annotationLoc)['bbox']
		bottom = bounding_box['bottom'][0][0][0][0]
		top = bounding_box['top'][0][0][0][0]
		left = bounding_box['left'][0][0][0][0]
		right = bounding_box['right'][0][0][0][0]
		bbox_image = self.image[top:bottom, left:right, :]
		return bbox_image
