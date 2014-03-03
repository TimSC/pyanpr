
import sys, os, deskew, pickle, deskewMarkedPlates
import scipy.misc as misc
import numpy as np
import skimage.exposure as exposure
import skimage.morphology as morph

def DeleteBlob(regionNum, blobIm):
	blobLocation = np.where(blobIm == regionNum)
	blobIm[blobLocation] = -1

def FindBlobs(scoreIm, numThresholds = 20, minLetterArea = 0.002, maxLetterArea = 0.05, bboxMinArea = 0.004, bboxMaxArea = 0.1, 
	minAspect = 1.0):

	normIm = exposure.rescale_intensity(scoreIm)

	thresholds = np.linspace(normIm.min(), normIm.max(), numThresholds)

	#Find blobs at various thresolds and calculate area
	thresholdBlobSizes = []
	thresholdBboxSizes = []
	numberedRegionIms = []
	bboxBrights = []
	for threshold in thresholds:

		thresholdIm = normIm < threshold

		blobSizes = []
		bboxSizes = []
		brightnessLi = []
		numberedRegions, numRegions = morph.label(thresholdIm, 4, 0, return_num = True)
		numberedRegionIms.append(numberedRegions)
		for regionNum in range(numRegions):
			#Gather statistics of blob
			regionIm = numberedRegions == regionNum
			bbox = BlobBbox(regionIm)
			blobSizes.append(regionIm.sum())
			
			bboxSizes.append(BlobBbox(regionIm))

			#Extract patch
			im2 = normIm[bbox[0]-1:bbox[1]+2,:]
			im3 = im2[:,bbox[2]-1:bbox[3]+2]

			stencil2 = regionIm[bbox[0]-1:bbox[1]+2,:]
			stencil3 = stencil2[:,bbox[2]-1:bbox[3]+2]

			notArea = np.logical_not(stencil3)
			areaVal = stencil3.sum()
			notAreaVal = notArea.sum()
			if areaVal != 0.:
				lo = (im3 * stencil3).sum() / areaVal
			else:
				lo = 0.5
			if notAreaVal != 0.:
				hi = (im3 * notArea).sum() / notAreaVal
			else:
				hi = 0.5

			brightnessLi.append((lo, hi))

			#print threshold, regionNum, regionIm.min(), regionIm.max(), regionIm.sum()

			
		thresholdBlobSizes.append(blobSizes)
		thresholdBboxSizes.append(bboxSizes)
		bboxBrights.append(brightnessLi)
	
#	for threshold, blobSizes, seedInRegion in zip(thresholds, thresholdBlobSizes, seedInRegionList):
#		blobSi = None
#		if seedInRegion != -1:
#			blobSi = blobSizes[seedInRegion]

#		print threshold, numRegions, seedInRegion, blobSi

		#thresholdIm = normIm < threshold
		#numberedRegions, maxRegionNum = morph.label(thresholdIm, 4, 0, return_num = True)
		#misc.imshow(numberedRegions==seedInRegion)

	bestThreshold = None
	bestRegionSum = None
	bestNumberedRegions = None
	bestThresholdBlobSizes = None

	#For each threshold
	for i, threshold in enumerate(thresholds):
		blobSizes = thresholdBlobSizes[i]
		bboxSizes = thresholdBboxSizes[i]
		numberedRegions = numberedRegionIms[i]
		brightnessLi = bboxBrights[i]

		#Sum the area of valid blobs
		count = 0
		validRegionNums = []
		validRegionSizes = []
		for regionNum, blobSize in enumerate(blobSizes):
			normBlobSize = float(blobSize) / normIm.size
			if normBlobSize < minLetterArea or normBlobSize > maxLetterArea:
				continue

			bbox = bboxSizes[regionNum]
			normBboxArea = float(bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) / bestNumberedRegions.size
			if normBboxArea < bboxMinArea or normBboxArea > bboxMaxArea:
				continue

			bboxAspect = float(bbox[1] - bbox[0]) / (bbox[3] - bbox[2])
			if bboxAspect < minAspect:
				continue

			count += 1
			validRegionNums.append(regionNum)
			validRegionSizes.append(blobSize)

		brightnessLi = np.array(brightnessLi)
		lo = 128.
		hi = 128.
		if brightnessLi.shape[0] > 0:
			lo, hi = brightnessLi[:,0].mean(), brightnessLi[:,1].mean()

		#Big areas are good
		areaSum = sum(validRegionSizes)

		#High contast separation is good
		contrastScore = (hi - lo)

		#The threshold should be between light and dark
		thresholdLoHiScore = 0.
		if (hi - lo) != 0.:
			thresholdLoHi = (threshold - lo) / (hi - lo)
			thresholdLoHiScore = 1. - abs(thresholdLoHi - 0.5)
			if thresholdLoHiScore < 0.:
				thresholdLoHiScore = 0.
		
		overallScore = areaSum * contrastScore * thresholdLoHiScore

		#print threshold, overallScore, areaSum, thresholdLoHi, lo, hi

		if bestThreshold is None or overallScore > bestRegionSum:
			bestRegionSum = overallScore
			bestThreshold = threshold
			bestNumberedRegions = numberedRegions
			bestThresholdBlobSizes = blobSizes

		#if len(validRegionSizes) > 0:
		#	import matplotlib.pyplot as plt
		#	plt.hist(validRegionSizes)
		#	plt.show()

		#if count > 0:
		#	print threshold, validRegionNums
		#	for vr in validRegionNums:
		#		misc.imshow(numberedRegions == vr)

	#Fliter blobs and remove any that are the wrong shape or size
	for i, blobSize in enumerate(bestThresholdBlobSizes):

		blobNormArea = float(blobSize) / bestNumberedRegions.size
		if blobNormArea > maxLetterArea or blobNormArea < minLetterArea:
			DeleteBlob(i, bestNumberedRegions)
			continue

		bbox = BlobBbox(bestNumberedRegions == i)
		bboxArea = float(bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) / bestNumberedRegions.size
		if bboxArea > bboxMaxArea or bboxArea < bboxMinArea:
			DeleteBlob(i, bestNumberedRegions)
			continue

		bboxAspect = float(bbox[1] - bbox[0]) / (bbox[3] - bbox[2])
		if bboxAspect < minAspect:
			DeleteBlob(i, bestNumberedRegions)
			continue
		
		#print i, blobNormArea, bboxArea, bboxAspect

	#Renumber regions
	outRegions, outNumRegions = morph.label(bestNumberedRegions, 4, -1, return_num = True)

	#print "best", bestThreshold, bestRegionSum
	return outRegions

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

def FitBboxModel(inlierBboxNum, bboxLi, imShape, numberedRegions, tolerance = 0.05):

	seedBbox = bboxLi[inlierBboxNum]

	#Find inliers for top and bottom edge
	inliers = []
 	for i, bbox in enumerate(bboxLi):
		topDiff = float(abs(seedBbox[0] - bbox[0])) / imShape[0]
		botDiff = float(abs(seedBbox[1] - bbox[1])) / imShape[0]
		if topDiff < tolerance and botDiff < tolerance:
			inliers.append(i)

	#print inliers
	bboxLi = np.array(bboxLi)
	inlierBboxes = bboxLi[inliers,:]

	robustTopY = inlierBboxes[:,0].mean()
	robustBottomY = inlierBboxes[:,1].mean()
	
	#Sort blobs by width
	blobWidth = [(bbox[3] - bbox[2], i) for (i, bbox) in enumerate(bboxLi)]
	blobWidth.sort()

	#For each blob, add to valid list if conditions are met
	blobWidths = []
	for blobWidth, blobNum in blobWidth[::-1]:
		bbox = bboxLi[blobNum]
		cofg = BlobCofG(numberedRegions == blobNum)
		#print blobWidth, blobNum, cofg
		if cofg[0] < robustTopY or cofg[0] > robustBottomY:
			continue #Outside lettering area

		#Check if this collides with existing width ranges
		containedWithin = []
		partialOverlapWith = []
		for i, (le, bw, cofgTmp) in enumerate(blobWidths):
			result = Check1DOverlap((bw[2], bw[3]), bbox[2:])
			if result == 1:
				partialOverlapWith.append(i)
			if result == 2:
				containedWithin.append(i)

		#if len(partialOverlapWith) > 0: continue
		if len(containedWithin) > 0: continue

		blobWidths.append((float(bbox[2]), bbox, cofg))

	blobWidths.sort(key=lambda x: x[0]) #Prevent sorting except on first key

	outBbox = []
	outCofG = []
	for leftEdge, bw, cofg in blobWidths:
		outBbox.append((bw[2], bw[3], bw[0], bw[1]))
		outCofG.append(cofg)
	#print len(out)
	return outBbox, outCofG

def FindCharacterBboxes(numberedRegions):
	
	maxRegion = numberedRegions.max()
	bboxLi = []
	for rn in range(maxRegion+1):
		regionIm = numberedRegions == rn
		bbox = BlobBbox(regionIm)
		bboxLi.append(bbox)

	#print maxRegion

	#Try each bounding box as a seed for model fitting
	#This is similar to ransac but we exhaustively try each starting bbox
	
	models = []
	maxModelSize = None
	maxModelSizeInd = None
	for seedRegionNum in range(maxRegion+1):

		if 0:
			regionIm = numberedRegions == seedRegionNum

			cb = bboxLi[seedRegionNum]
			im2 = regionIm[cb[0]:cb[1]+1,:]
			im3 = im2[:,cb[2]:cb[3]+1,]
			if im3.size > 0:
				misc.imshow(im3)
	
		model = FitBboxModel(seedRegionNum, bboxLi, numberedRegions.shape, numberedRegions)
		models.append(model)
		if maxModelSize is None or len(model) > maxModelSize:
			maxModelSize = len(model)
			maxModelSizeInd = seedRegionNum

		#print len(model)
	
	#for model in models:
	#	print len(model)

	#Return biggest model
	#TODO find a better way to select the best
	if maxModelSizeInd is None:
		return None, None
	return models[maxModelSizeInd]

def DetectCharacters(im):
	bestNumberedRegions = FindBlobs(im)

	numberedRegionsIm = exposure.rescale_intensity(bestNumberedRegions != -1)
	#misc.imshow(numberedRegionsIm)

	charBboxes, charCofG = FindCharacterBboxes(bestNumberedRegions)
	return charBboxes, charCofG

if __name__ == "__main__":
	finaIm = None
	finaDeskew = None

	if len(sys.argv) >= 2:
		finaIm = sys.argv[1]
	if len(sys.argv) >= 3:
		finaDeskew = sys.argv[2]

	finaImSplitPath = os.path.split(finaIm)
	finaImSplitExt = os.path.splitext(finaImSplitPath[1])
	if finaDeskew is None:
		finaDeskew = "train/" + finaImSplitExt[0] +".deskew"
	if finaDeskew is None or not os.path.isfile(finaDeskew):
		finaDeskew = finaImSplitPath[0] + "/" +finaImSplitExt[0] +".deskew"

	im = misc.imread(finaIm)
	bbox, angle = pickle.load(open(finaDeskew))
	
	rotIm = deskew.RotateAndCrop(im, bbox, angle)
	
	scoreIm = deskewMarkedPlates.RgbToPlateBackgroundScore(rotIm)
	charBboxes, charCofG = DetectCharacters(scoreIm)

	mergedChars = None
	sepImg = None

	for cb in charBboxes:
		im2 = rotIm[cb[2]:cb[3]+1,:,:]
		im3 = im2[:,cb[0]:cb[1]+1,:]
		if mergedChars is None:
			mergedChars = im3
			sepImg = np.ones((im3.shape[0], 10, 3)) * 0.5
		else:
			if im3.shape[0] > mergedChars.shape[0]:
				old = mergedChars
				mergedChars = np.zeros((im3.shape[0], mergedChars.shape[1], 3), dtype=old.dtype)
				mergedChars[:old.shape[0], :old.shape[1], :] = old
				sepImg = np.ones((im3.shape[0], 10, 3)) * 0.5
			if im3.shape[0] < mergedChars.shape[0]:
				old = im3
				im3 = np.zeros((mergedChars.shape[0], im3.shape[1], 3), dtype=old.dtype)
				im3[:old.shape[0], :old.shape[1], :] = old

			mergedChars = np.hstack((mergedChars, sepImg, im3))

	misc.imshow(mergedChars)

