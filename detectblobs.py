
import sys, os, deskew, pickle, deskewMarkedPlates
import scipy.misc as misc
import numpy as np
import skimage.exposure as exposure
import skimage.morphology as morph

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

	normIm = exposure.rescale_intensity(scoreIm)

	thresholds = np.linspace(normIm.min(), normIm.max(), 10.)

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
	stabilityIm = np.empty(normIm.shape, dtype=np.uint8)
	
	for x in range(normIm.shape[1]):
		for y in range(normIm.shape[0]):
			areaProfile = []
			for threshold, blobSizes, numberedRegions in zip(thresholds, thresholdBlobSizes, numberedRegionIms):
				ptInRegion = numberedRegions[y, x]
				#print x, y, ptInRegion
				if ptInRegion != -1:
					areaProfile.append(blobSizes[ptInRegion])
				else:
					areaProfile.append(None)

			#print x, y, areaProfile, numberedRegions.size

			scoringThresholds = 0
			for val in areaProfile:
				if val is None: continue
				if val > 0.3 * numberedRegions.size: continue
				scoringThresholds += 1

			#print x, y, scoringThresholds

			stabilityIm[y, x] = scoringThresholds

	stabilityIm = exposure.rescale_intensity(stabilityIm)
	misc.imshow(stabilityIm)

