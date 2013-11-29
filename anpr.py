
import scipy.misc as misc
import scipy.signal as signal
import numpy as np
import skimage.morphology as morph
import matplotlib.pyplot as plt
import math, sys, pickle, os

def ScoreUsingAspect(numberedRegions, vis = None):
	#Use first criterion (region aspect ratio) to select candidates
	maxRegionNum = numberedRegions.max()

	regionScores = []
	for regionNum in range(maxRegionNum):

		region = (numberedRegions == regionNum) #Isolate region
		pixPos = np.where(region == True)
		bbox = [[pixPos[1].min(), pixPos[1].max()], [pixPos[0].min(), pixPos[0].max()]]
		xr = pixPos[0].max() - pixPos[0].min()
		yr = pixPos[1].max() - pixPos[1].min()
		if xr != 0.:
			aspect = float(yr) / float(xr)
		else:
			aspect = 0.
		area = xr * yr

		targetAspect = 6.8
		aspectErr = abs(targetAspect - aspect)
		if aspectErr > 1e-6:
			score = 1. / aspectErr
		else:
			score = 1e-6
		print regionNum, score, aspect, aspectErr

		regionScores.append((score, regionNum, bbox))

	if vis is not None:
		maxScore = max([i[0] for i in regionScores])
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
		bbox = [[pixPos[1].min(), pixPos[1].max()], [pixPos[0].min(), pixPos[0].max()]]
		yr = pixPos[0].max() - pixPos[0].min()
		xr = pixPos[1].max() - pixPos[1].min()
		area = xr * yr

		xwtarget = 0.34
		ywtarget = 0.07
		xw = float(xr)/imshape[1]
		yw = float(yr)/imshape[0]
		xerr = abs(xw - xwtarget)
		yerr = abs(yw - ywtarget)
		if xerr < 0.001:
			xerr = 0.001
		if yerr < 0.001:
			yerr = 0.001
		score = (1. / xerr) * (1. / yerr)
		print regionNum, xw, yw, score
		regionScores2.append((score, regionNum, bbox))

	if vis is not None:
		maxScore = max([i[0] for i in regionScores2])
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
	#IET Circuits Devices Syst., 2013, Vol. 7, Iss. 2, pp. 93-103

	fina = None
	if len(sys.argv) >= 2:
		fina = sys.argv[1]
	if fina is None:
		print "Specify input image on command line"
		exit(0)

	im = misc.imread(fina)
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]

	#Highlight number plate
	imnorm = np.array(greyim, dtype=np.uint8)
	se = np.ones((3, 30), dtype=np.uint8)
	opim = morph.opening(imnorm, se)
	diff = greyim - opim + 128.

	misc.imsave("diff.png", diff)

	#Binarize image
	vals = diff.copy()
	vals = vals.reshape((vals.size))

	meanVal = vals.mean()
	stdVal = vals.std()
	threshold = meanVal + stdVal

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

	if not os.path.exists("candidates"):
		os.mkdir("candidates")

	scores1 = ScoreUsingAspect(numberedRegions, "firstcritera.png")
	scores1.sort()
	scores1.reverse()
	print "Using first criteria", scores1[0]

	for i, can in enumerate(scores1):
		bbox = can[2]
		patchIm = im[bbox[1][0]:bbox[1][1],:]
		patchIm = patchIm[:,bbox[0][0]:bbox[0][1]]
		misc.imsave("candidates/1-{0}.png".format(i), patchIm)
		pickle.dump(can, open("candidates/1-{0}.dat".format(i), "wb"), protocol=-1)

	scores2 = ScoreUsingSize(numberedRegions, binIm.shape, "secondcriteria.png")
	scores2.sort()
	scores2.reverse()
	print "Using second criteria", scores2[0]

	for i, can in enumerate(scores2):
		bbox = can[2]
		patchIm = im[bbox[1][0]:bbox[1][1],:]
		patchIm = patchIm[:,bbox[0][0]:bbox[0][1]]
		misc.imsave("candidates/1-{0}.png".format(i), patchIm)
		pickle.dump(can, open("candidates/1-{0}.dat".format(i), "wb"), protocol=-1)

	


	

	

