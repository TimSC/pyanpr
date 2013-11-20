
import scipy.misc as misc
import scipy.signal as signal
import numpy as np
import skimage.morphology as morph
import matplotlib.pyplot as plt
import math, sys

def ScoreUsingAspect(numberedRegions, vis = None):
	#Use first criterion (region aspect ratio) to select candidates
	maxRegionNum = numberedRegions.max()

	regionScores = []
	for regionNum in range(maxRegionNum):

		region = (numberedRegions == regionNum) #Isolate region
		pixPos = np.where(region == True)
		xr = pixPos[0].max() - pixPos[0].min()
		yr = pixPos[1].max() - pixPos[1].min()
		if xr != 0.:
			aspect = float(yr) / float(xr)
		else:
			aspect = 0.
		area = xr * yr

		targetAspect = 4.6
		aspectErr = abs(targetAspect - aspect)
		if aspectErr > 1e-6:
			score = 1. / aspectErr
		else:
			score = 1e-6
		#print regionNum, score, aspect, aspectErr

		regionScores.append((score, regionNum))

	if vis is not None:
		maxScore = np.array(regionScores)[:,0].max()
		visCandidates = np.zeros(binIm.shape)
		for regionNum, score in enumerate(regionScores):
			region = (numberedRegions == regionNum) #Isolate region

			visCandidates += region * (score[0] / maxScore) * 255.
			misc.imsave(vis, visCandidates)

	return regionScores

def ScoreUsingSize(numberedRegions, imshape, vis = None):
	#Use second criteria (of x and y range of region) to select candidates
	maxRegionNum = numberedRegions.max()

	regionScores2 = []
	for regionNum in range(maxRegionNum):

		region = (numberedRegions == regionNum) #Isolate region
		pixPos = np.where(region == True)
		xr = pixPos[0].max() - pixPos[0].min()
		yr = pixPos[1].max() - pixPos[1].min()
		area = xr * yr

		xwtarget = 0.06
		ywtarget = 0.2
		xw = float(xr)/imshape[0]
		yw = float(yr)/imshape[1]
		xerr = abs(xw - xwtarget)
		yerr = abs(yw - ywtarget)
		if xerr < 0.001:
			xerr = 0.001
		if yerr < 0.001:
			yerr = 0.001
		score = (1. / xerr) * (1. / yerr)
		regionScores2.append((score, regionNum))

	if vis is not None:
		maxScore = np.array(regionScores2)[:,0].max()
		visCandidates = np.zeros(binIm.shape)
		for regionNum, score in enumerate(regionScores2):
			region = (numberedRegions == regionNum) #Isolate region

			visCandidates += region * (score[0] / maxScore) * 255.
			misc.imsave(vis, visCandidates)

	return regionScores2

if __name__ == "__main__":

	#Improved number plate localisation algorithm and its efficient 
	#field programmable gate arrays implementation
	#Xiaojun Zhai, Faycal Bensaali, Soodamani Ramalingam

	fina = None
	if len(sys.argv) >= 2:
		fina = sys.argv[1]
	if fina is None:
		print "Specify input image on command line"
		exit(0)

	im = misc.imread(fina)
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]

	#Highlight number plate
	imnorm = greyim / 255.
	se = np.ones((3, 30))
	opim = morph.opening(imnorm, se)

	diff = greyim - opim + 128.

	misc.imsave("diff.png", diff)

	#Binarize image
	vals = diff.copy()
	vals = vals.reshape((vals.size))
	vals.sort()
	ind = int(round(vals.size * 0.98))
	print "Index", ind, "of", vals.size 
	threshold = vals[ind]
	print vals[ind-1], vals[ind], vals[ind+1]
	print "Threshold", threshold

	binIm = diff > threshold
	misc.imsave("threshold.png", binIm)
	#print vals.shape
	#plt.plot(vals)
	#plt.show()

	#Denoise
	diamond = morph.diamond(2)
	denoiseIm = morph.binary_opening(binIm, diamond)
	denoiseIm2 = morph.binary_closing(denoiseIm, np.ones((3, 13)))

	#Number candidate regions
	print "Numbering regions"
	numberedRegions, maxRegionNum = morph.label(denoiseIm2, 4, 0, return_num = True)

	scores1 = ScoreUsingAspect(numberedRegions, "firstcritera.png")
	print "Using first criteria", scores1[-1]

	scores2 = ScoreUsingSize(numberedRegions, binIm.shape, "secondcriteria.png")
	print "Using second criteria", scores2[-1]

	


	

	

