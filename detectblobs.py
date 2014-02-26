
import sys, os, deskew, pickle, deskewMarkedPlates
import scipy.misc as misc
import numpy as np
import skimage.exposure as exposure
import skimage.morphology as morph

def FindBlobs(scoreIm, numThresholds = 100, maxLetterArea = 0.05):
	normIm = exposure.rescale_intensity(scoreIm)

	thresholds = np.linspace(normIm.min(), normIm.max(), numThresholds)

	#Find blobs at various thresolds and calculate area
	thresholdBlobSizes = []
	numberedRegionIms = []
	for threshold in thresholds:

		thresholdIm = normIm < threshold

		blobSizes = []
		numberedRegions, maxRegionNum = morph.label(thresholdIm, 4, 0, return_num = True)
		numberedRegionIms.append(numberedRegions)
		for regionNum in range(maxRegionNum):
			regionIm = numberedRegions == regionNum
			#print threshold, regionNum, regionIm.min(), regionIm.max(), regionIm.sum()
			blobSizes.append(regionIm.sum())
		thresholdBlobSizes.append(blobSizes)
	
#	for threshold, blobSizes, seedInRegion in zip(thresholds, thresholdBlobSizes, seedInRegionList):
#		blobSi = None
#		if seedInRegion != -1:
#			blobSi = blobSizes[seedInRegion]

#		print threshold, maxRegionNum, seedInRegion, blobSi

		#thresholdIm = normIm < threshold
		#numberedRegions, maxRegionNum = morph.label(thresholdIm, 4, 0, return_num = True)
		#misc.imshow(numberedRegions==seedInRegion)

	bestThreshold = None
	bestRegionSum = None
	bestNumberedRegions = None

	for threshold, blobSizes, numberedRegions in zip(thresholds, thresholdBlobSizes, numberedRegionIms):
		count = 0
		validRegionNums = []
		validRegionSizes = []
		for regionNum, blobSize in enumerate(blobSizes):
			if float(blobSize) / normIm.size < maxLetterArea:
				count += 1
				validRegionNums.append(regionNum)
				validRegionSizes.append(blobSize)

		#print threshold, count, len(blobSizes), validRegionSizes
		
		areaSum = sum(validRegionSizes)
		if bestThreshold is None or areaSum > bestRegionSum:
			bestRegionSum = areaSum
			bestThreshold = threshold
			bestNumberedRegions = numberedRegions

		#if len(validRegionSizes) > 0:
		#	import matplotlib.pyplot as plt
		#	plt.hist(validRegionSizes)
		#	plt.show()

		#if count > 0:
		#	print threshold, validRegionNums
		#	for vr in validRegionNums:
		#		misc.imshow(numberedRegions == vr)

	print "best", bestThreshold, bestRegionSum
	return bestNumberedRegions


if __name__ == "__main__":
	finaIm = None
	finaDeskew = None

	if len(sys.argv) >= 2:
		finaIm = sys.argv[1]
	if len(sys.argv) >= 3:
		finaDeskew = sys.argv[2]

	finaImSplit = os.path.splitext(finaIm)
	if finaDeskew is None:
		finaDeskew = finaImSplit[0] +".deskew"

	im = misc.imread(finaIm)
	bbox, angle = pickle.load(open(finaDeskew))
	

	rotIm = deskew.RotateAndCrop(im, bbox, angle)
	
	scoreIm = deskewMarkedPlates.RgbToPlateBackgroundScore(rotIm)

	bestNumberedRegions = FindBlobs(scoreIm)

	numberedRegionsIm = exposure.rescale_intensity(bestNumberedRegions != -1)
	misc.imshow(numberedRegionsIm)

