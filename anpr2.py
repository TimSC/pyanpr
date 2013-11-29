import pickle, sys, math
import scipy.misc as misc
import skimage.filter as filt
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

if __name__=="__main__":

	#Deskew the candidate patch
	#Based on ALGORITHMIC AND MATHEMATICAL PRINCIPLES OF AUTOMATIC NUMBER PLATE RECOGNITION SYSTEMS 
	#ONDREJ MARTINSKY

	finaIm = None
	finaDat = None

	if len(sys.argv) >= 2:
		finaIm = sys.argv[1]
	if len(sys.argv) >= 3:
		finaDat = sys.argv[2]

	if finaIm is None or finaDat is None:
		print "Specify input image and data on command line (2 args)"
		exit(0)
	im = misc.imread(finaIm)

	roi = pickle.load(open(finaDat,"rb"))
	bbox = roi[2]
	print bbox

	#Expand bbox
	mid = [sum(comp)/len(comp) for comp in bbox]
	print "mid", mid
	xd = (mid[0] - bbox[0][0]) * 0.2
	yd = (mid[1] - bbox[1][0]) * 0.2
	bbox = [[bbox[0][0]-xd, bbox[0][1]+xd], [bbox[1][0]-yd, bbox[1][1]+yd]]
	print bbox

	#Crop image
	im = im[bbox[1][0]:bbox[1][1],:]
	im = im[:,bbox[0][0]:bbox[0][1]]
	print im.shape

	#Convert to grey
	misc.imsave("test.png", im)
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]

	edgeIm = scipy.ndimage.sobel(greyim, axis=0)
	misc.imsave("test2.png", edgeIm)
	edgeIm = np.abs(edgeIm.astype(np.float)-128.)
	misc.imsave("test3.png", edgeIm)

	
	midThresh = 0.5 * (edgeIm.min() + edgeIm.max())
	thresholdIm = (edgeIm > midThresh)
	misc.imsave("test4.png", thresholdIm)

	h, theta, d = hough_line(thresholdIm)

	if 0:
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

	print bestInd, bestAngle, math.degrees(bestAngle)
	pickle.dump((bbox, bestAngle), open("out.deskew", "wb"), protocol=-1)


