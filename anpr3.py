#Recognise number plate letters

import pickle, sys, math, os
import scipy.misc as misc
import skimage.transform as transform
import numpy as np
import skimage.morphology as morph

if __name__=="__main__":
	
	finaIm = None
	finaSkew = None

	if len(sys.argv) >= 2:
		finaIm = sys.argv[1]
	if len(sys.argv) >= 3:
		finaSkew = sys.argv[2]
	
	if finaIm is None:
		print "Specify input image on command line"
		exit(0)
	finaImSplit = os.path.splitext(finaIm)
	if finaSkew is None:
		finaSkew = finaImSplit[0] +".deskew"

	im = misc.imread(finaIm)

	skew = pickle.load(open(finaSkew,"rb"))
	bbox = skew[0]
	shewAng = skew[1]
	print "skew", skew

	#Crop input image
	im = im[:, bbox[0][0]:bbox[0][1]]
	im = im[bbox[1][0]:bbox[1][1], :]

	#Convert to greyscale
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]
	misc.imsave("test.png", greyim)

	#Rotate image
	rotIm = transform.rotate(greyim.astype(np.uint8), math.degrees(shewAng))
	misc.imsave("test2.png", rotIm)

	#Threshold
	midIntensity = (rotIm.max() - rotIm.min()) * 0.5
	binIm = rotIm < midIntensity
	misc.imsave("test3.png", binIm)

	#Number regions
	numberedRegions, maxRegionNum = morph.label(binIm, 4, 0, return_num = True)

	#For each numbered region
	for i in range(maxRegionNum):
		region = (numberedRegions == i)
		pixloc = np.where(region)
		miny = pixloc[0].min()
		maxy = pixloc[0].max()
		minx = pixloc[1].min()
		maxx = pixloc[1].max()
		print (minx, maxx), (miny, maxy)

		#Crop letter
		letterIm = rotIm[miny:maxy+1,:]
		letterIm = letterIm[:,minx:maxx+1]
		print letterIm.shape

		misc.imsave("letter{0}.png".format(i), letterIm)



