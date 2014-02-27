
import sys, os, deskew, pickle, deskewMarkedPlates
import scipy.misc as misc
import numpy as np
import skimage.exposure as exposure
import skimage.morphology as morph

def DeleteBlob(regionNum, blobIm):
	blobLocation = np.where(blobIm == regionNum)
	blobIm[blobLocation] = -1

def FindBlobs(scoreIm, numThresholds = 100, minLetterArea = 0.002, maxLetterArea = 0.05, bboxMinArea = 0.004, bboxMaxArea = 0.1, 
	minAspect = 0.5):

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
	bestThresholdBlobSizes = None

	for i, (threshold, blobSizes, numberedRegions) in enumerate(zip(thresholds, thresholdBlobSizes, numberedRegionIms)):
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
			bestThresholdBlobSizes = thresholdBlobSizes[i]

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
	outRegions, outMaxRegion = morph.label(bestNumberedRegions, 4, -1, return_num = True)

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

		#print "a", blobNum
		#Check if this collides with existing width ranges
		containedWithin = []
		partialOverlapWith = []
		for i, bw in enumerate(blobWidths):
			result = Check1DOverlap(bw, bbox[2:])
			if result == 1:
				partialOverlapWith.append(i)
			if result == 2:
				containedWithin.append(i)

		#print "b", blobNum, partialOverlapWith, containedWithin
		if len(partialOverlapWith) > 0: continue
		if len(containedWithin) > 0: continue
		#print "c", blobNum

		blobWidths.append((bbox[2], bbox[3]))

	blobWidths.sort()

	out = []
	for bw in blobWidths:
		out.append(bw + (robustTopY, robustBottomY))
	#print len(out)
	return out
		
def FindCharacterBboxes(numberedRegions):
	
	maxRegion = numberedRegions.max()
	bboxLi = []
	for rn in range(maxRegion):
		regionIm = numberedRegions == rn
		bbox = BlobBbox(regionIm)
		bboxLi.append(bbox)

	#Try each bounding box as a seed for model fitting
	#This is similar to ransac but we exhaustively try each starting bbox
	
	models = []
	maxModelSize = None
	maxModelSizeInd = None
	for seedRegionNum in range(maxRegion):

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
	return models[maxModelSizeInd]

def DetectCharacters(im):
	bestNumberedRegions = FindBlobs(im)

	numberedRegionsIm = exposure.rescale_intensity(bestNumberedRegions != -1)
	misc.imshow(numberedRegionsIm)

	charBboxes = FindCharacterBboxes(bestNumberedRegions)
	return charBboxes

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
	charBboxes = DetectCharacters(scoreIm)

	print charBboxes

	for cb in charBboxes:
		print cb
		print rotIm.shape
		im2 = rotIm[cb[2]:cb[3]+1,:,:]
		im3 = im2[:,cb[0]:cb[1]+1,:]
		misc.imshow(im3)


