
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

	seedPoint = 150, 50

	thresholds = np.linspace(normIm.min(), normIm.max(), 10.)

	thresholdBlobSizes = []
	seedInRegionList = []
	for threshold in thresholds:

		thresholdIm = normIm < threshold

		blobSizes = []
		numberedRegions, maxRegionNum = morph.label(thresholdIm, 4, 0, return_num = True)
		for regionNum in range(maxRegionNum):
			regionIm = numberedRegions == regionNum
			#print threshold, regionNum, regionIm.min(), regionIm.max(), regionIm.sum()
			blobSizes.append(regionIm.sum())
		thresholdBlobSizes.append(blobSizes)

		seedInRegion = numberedRegions[seedPoint[1], seedPoint[0]]
		seedInRegionList.append(seedInRegion)

	for threshold, blobSizes, seedInRegion in zip(thresholds, thresholdBlobSizes, seedInRegionList):
		blobSi = None
		if seedInRegion != -1:
			blobSi = blobSizes[seedInRegion]

		print threshold, maxRegionNum, seedInRegion, blobSi

		thresholdIm = normIm < threshold
		numberedRegions, maxRegionNum = morph.label(thresholdIm, 4, 0, return_num = True)
		misc.imshow(numberedRegions==seedInRegion)

