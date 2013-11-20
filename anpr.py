
import scipy.misc as misc
import scipy.signal as signal
import numpy as np
import skimage.morphology as morph
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":

	#Improved number plate localisation algorithm and its efficient 
	#field programmable gate arrays implementation
	#Xiaojun Zhai, Faycal Bensaali, Soodamani Ramalingam

	im = misc.imread("../pyanpr-data/56897161_d613d63bce_b.jpg")
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]

	#Highlight number plate
	imnorm = greyim / 255.
	se = np.ones((3, 30))
	opim = morph.opening(imnorm, se)

	diff = greyim - opim + 128.

	#Binarize image
	vals = diff.copy()
	vals = vals.reshape((vals.size, 1))
	vals.sort()
	threshold = vals[math.floor(vals.size * 0.98)]
	print "Threshold", threshold

	#plt.hist(vals, bins=100)
	#plt.show()

	binIm = diff > threshold

	#Denoise
	diamond = morph.diamond(2)
	denoiseIm = morph.binary_opening(binIm, diamond)
	denoiseIm2 = morph.binary_closing(denoiseIm, np.ones((3, 13)))

	#Number candidate regions
	numberedRegions, maxRegionNum = morph.label(denoiseIm2, 4, 0, return_num = True)

	#Use first criterion (region aspect ratio) to select candidates
	regionScores = []
	for regionNum in range(maxRegionNum):

		region = (numberedRegions == regionNum) #Isolate region
		pixPos = np.where(region == True)
		xr = pixPos[0].max() - pixPos[0].min()
		yr = pixPos[1].max() - pixPos[1].min()
		if yr != 0.:
			aspect = float(xr) / float(yr)
		else:
			aspect = 0.
		area = xr * yr

		targetAspect = 0.21
		score = (area ** 0.5) * (1.-abs(targetAspect - aspect))
		#print regionNum, score, aspect

		regionScores.append((score, regionNum))

	maxScore = np.array(regionScores)[:,0].max()

	visCandidates = np.zeros(binIm.shape)
	for regionNum, score in enumerate(regionScores):
		region = (numberedRegions == regionNum) #Isolate region

		visCandidates += region * (score[0] / maxScore) * 255.
		

	misc.imsave("opim.png", visCandidates)

	

