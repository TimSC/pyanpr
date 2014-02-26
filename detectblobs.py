
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

def BlobBbox(im):
	pos = np.where(im==True)
	return pos[0].min(), pos[0].max(), pos[1].min(), pos[1].max()

def BlobCofG(im):
	pos = np.where(im==True)
	return pos[0].mean(), pos[1].mean()

def Check1DOverlap(range1, range2):
	min1 = min(range1)
	max1 = max(range1)
	min2 = min(range2)
	max2 = max(range2)
	#print min1, max1, min2, max2
	if min1 >= min2 and min1 <= max2: return 1 #Partial overlap
	if max1 >= min2 and max1 <= max2: return 1 #Partial overlap
	if min1 <= min2 and max1 >= max2: return 2 #Contained
	return False

def FindCharacterBboxes(numberedRegions):
	
	maxRegion = numberedRegions.max()
	bboxLi = []
	for rn in range(maxRegion):
		regionIm = numberedRegions == rn
		bbox = BlobBbox(regionIm)
		bboxLi.append(bbox)

	#The tops and bottoms of the letters are aligned
	bboxLi = np.array(bboxLi)
	topY = bboxLi[:,0]
	bottomY = bboxLi[:,1]
	medianTopY = int(round(np.median(topY)))
	medianBottomY = int(round(np.median(bottomY)))

	print "median top and bottom", medianTopY, medianBottomY
	
	#Sort blobs by width
	blobWidth = [(bbox[3] - bbox[2], i) for (i, bbox) in enumerate(bboxLi)]
	blobWidth.sort()

	#For each blob, add to valid list if conditions are met
	blobWidths = []
	for blobWidth, blobNum in blobWidth[::-1]:
		bbox = bboxLi[blobNum]
		cofg = BlobCofG(numberedRegions == blobNum)
		#print blobWidth, blobNum, cofg
		if cofg[0] < medianTopY or cofg[0] > medianBottomY:
			continue #Outside lettering area

		#Check if this collides with existing width ranges
		containedWithin = []
		partialOverlapWith = []
		for i, bw in enumerate(blobWidths):
			result = Check1DOverlap(bw, bbox[2:])
			if result == 1:
				partialOverlapWith.append(i)
			if result == 2:
				containedWithin.append(i)

		if len(partialOverlapWith) > 0: continue
		if len(containedWithin) > 0: continue

		blobWidths.append((bbox[2], bbox[3]))

	blobWidths.sort()

	out = []
	for bw in blobWidths:
		out.append(bw + (medianTopY, medianBottomY))
	return out

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

	#numberedRegionsIm = exposure.rescale_intensity(bestNumberedRegions != -1)
	#misc.imshow(numberedRegionsIm)

	charBboxes = FindCharacterBboxes(bestNumberedRegions)
	print charBboxes


