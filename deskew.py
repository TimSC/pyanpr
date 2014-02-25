import pickle, sys, math, os
import scipy.misc as misc
import skimage.filter as filt
import skimage.transform as transform
from skimage.transform import hough_line, hough_line_peaks
import skimage.transform as transform
import numpy as np
import scipy.ndimage

def ExpandBBox(bbox, factor):
	print "bbox", bbox
	#Expand bbox
	mid = [sum(comp)/len(comp) for comp in bbox]
	print "mid", mid
	xd = (mid[0] - bbox[0][0]) * factor
	yd = (mid[1] - bbox[1][0]) * factor
	bboxMod = [[bbox[0][0]-xd, bbox[0][1]+xd], [bbox[1][0]-yd, bbox[1][1]+yd]]
	bboxMod = [map(int, map(round, pt)) for pt in bboxMod]
	print "bbox expanded", bboxMod
	return bboxMod

def CropToBBox(im, bbox):
	im2 = im[bbox[1][0]:bbox[1][1],:]
	return im2[:,bbox[0][0]:bbox[0][1]]

def Deskew(im, bbox, saveTempImages = False):
	#Deskew the candidate patch
	#Based on ALGORITHMIC AND MATHEMATICAL PRINCIPLES OF AUTOMATIC NUMBER PLATE RECOGNITION SYSTEMS 
	#ONDREJ MARTINSKY

	#im is a numpy array, with 3 dimensions RGB

	#bbox is a list of tuples:
	#[(min_x, max_x), (min_y, max_y)]
	#Example: [(735.4, 1299.24), (1296.0, 1399.6)]

	bboxMod = ExpandBBox(bbox, 0.2)
	im = CropToBBox(im, bboxMod)

	#Crop image to get ROI
	print "im original shape", im.shape

	print "im shape", im.shape

	#Convert to grey
	if saveTempImages:
		misc.imsave("test.png", im)
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]

	edgeIm = scipy.ndimage.sobel(greyim, axis=0)
	if saveTempImages:
		misc.imsave("test2.png", edgeIm)
	edgeIm = np.abs(edgeIm.astype(np.float)-128.)
	if saveTempImages:
		misc.imsave("test3.png", edgeIm)

	midThresh = 0.5 * (edgeIm.min() + edgeIm.max())
	thresholdIm = (edgeIm > midThresh)
	if saveTempImages:
		misc.imsave("test4.png", thresholdIm)

	h, theta, d = hough_line(thresholdIm)

	if 0:
		import matplotlib.pyplot as plt
		plt.imshow(np.log(h+1),
		       extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
		               d[-1], d[0]],
		       cmap=plt.cm.gray, aspect=1/1.5)
		plt.title('Hough transform')
		plt.xlabel('Angles (degrees)')
		plt.ylabel('Distance (pixels)')
		plt.show()
	
	angsum = np.sum(np.power(h, 2.0),axis=0)
	#print angsum
	#plt.plot(angsum)
	#plt.show()
	bestInd = angsum.argmax()
	bestAngle = theta[bestInd]

	while bestAngle > math.pi / 4.:
		bestAngle -= math.pi /2.
	while bestAngle < -math.pi /4.:
		bestAngle += math.pi /2.

	return bbox, bestInd, bestAngle

def RotateAndCrop(im, bbox, bestAngle):

	pat = transform.PiecewiseAffineTransform()
	print "test", bbox	

	dstPts = [(bbox[0][0],bbox[1][0]),
		(bbox[0][1],bbox[1][0]),
		(bbox[0][1],bbox[1][1]),
		(bbox[0][0],bbox[1][1])]

	srcPts = [(0.,0.),
		(bbox[0][1]-bbox[0][0],0.),
		(bbox[0][1]-bbox[0][0],bbox[1][1]-bbox[1][0]),
		(0.,bbox[1][1]-bbox[1][0])]

	srcPts = np.array(srcPts)
	dstPts = np.array(dstPts)

	#Rotate points
	#TODO

	pat.estimate(srcPts, dstPts)
	return transform.warp(im, pat, output_shape=(bbox[1][1]-bbox[1][0], bbox[0][1]-bbox[0][0]))

if __name__=="__main__":

	finaIm = None
	finaDat = None
	finaOut = None

	if len(sys.argv) >= 2:
		finaIm = sys.argv[1]
	if len(sys.argv) >= 3:
		finaDat = sys.argv[2]
	if len(sys.argv) >= 4:
		finaOut = sys.argv[3]

	if finaIm is None:
		print "Specify input image on command line"
		exit(0)
	finaImSplit = os.path.splitext(finaIm)
	if finaDat is None:
		finaDat = finaImSplit[0] +".dat"
	if finaOut is None:
		finaOut = finaImSplit[0] +".deskew"

	im = misc.imread(finaIm)

	roi = pickle.load(open(finaDat,"rb"))
	bbox = roi[2]
	print "bbox", bbox

	bbox, bestInd, bestAngle = Deskew(im, bbox, True)

	print bestInd, bestAngle, math.degrees(bestAngle)
	pickle.dump((bbox, bestAngle), open(finaOut, "wb"), protocol=-1)

	rotIm = RotateAndCrop(im, bbox, bestAngle)

	misc.imsave("rotIm.png", rotIm)

