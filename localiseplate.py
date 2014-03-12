
#Based on "Improved number plate localisation algorithm and its efficient 
#field programmable gate arrays implementation"
#Xiaojun Zhai, Faycal Bensaali, Soodamani Ramalingam
#IET Circuits Devices Syst., 2013, Vol. 7, Iss. 2, pp. 93-103

import scipy.misc as misc
import scipy.signal as signal
import numpy as np
import skimage.morphology as morph
import matplotlib.pyplot as plt
import math, sys, pickle, os

def ExtractPatch(image, bbox):
	bbox = map(int,map(round,bbox))
	#print image.shape, bbox
	out = np.zeros((bbox[3]-bbox[2], bbox[1]-bbox[0], 3), dtype=image.dtype)

	origin = [0, 0]
	if bbox[0] < 0:
		origin[0] = -bbox[0]
		bbox[0] = 0
	if bbox[2] < 0:
		origin[1] = -bbox[2]
		bbox[2] = 0
	if bbox[1] >= image.shape[1]:
		bbox[1] = image.shape[1]-1
	if bbox[3] >= image.shape[0]:
		bbox[3] = image.shape[0]-1
	
	h = bbox[3]-bbox[2]
	w = bbox[1]-bbox[0]

	#print bbox
	out[origin[1]:origin[1]+h, origin[0]:origin[0]+w, :] = image[bbox[2]:bbox[2]+h, bbox[0]:bbox[0]+w, :]
	return out

def ScoreUsingAspect(numberedRegions, targetAspect = 4.0, vis = None):
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

		aspectErr = abs(targetAspect - aspect)
		if aspectErr > 1e-6:
			score = 1. / aspectErr
		else:
			score = 1e-6
		#print regionNum, score, aspect, aspectErr

		regionScores.append([score, regionNum, bbox])

	if vis is not None:
		maxScore = max([i[0] for i in regionScores])
		visCandidates = np.zeros(numberedRegions.shape)
		for regionNum, score in enumerate(regionScores):
			region = (numberedRegions == regionNum) #Isolate region

			visCandidates += region * (score[0] / maxScore) * 255.
			misc.imsave(vis, visCandidates)

	regionScores.sort()
	regionScores.reverse()

	return regionScores

def ScoreUsingSize(numberedRegions, xwtarget = 0.14, ywtarget = 0.03, vis = None):
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

		xw = float(xr)/numberedRegions.shape[1]
		yw = float(yr)/numberedRegions.shape[0]
		xerr = abs(xw - xwtarget)
		yerr = abs(yw - ywtarget)
		if xerr < 0.001:
			xerr = 0.001
		if yerr < 0.001:
			yerr = 0.001
		score = (1. / xerr) * (1. / yerr)
		#print regionNum, xw, yw, score
		regionScores2.append([score, regionNum, bbox])

	if vis is not None:
		maxScore = max([i[0] for i in regionScores2])
		visCandidates = np.zeros(numberedRegions.shape)
		for regionNum, score in enumerate(regionScores2):
			region = (numberedRegions == regionNum) #Isolate region

			visCandidates += region * (score[0] / maxScore) * 255.
			misc.imsave(vis, visCandidates)

	regionScores2.sort()
	regionScores2.reverse()

	return regionScores2

def ProcessImage(im, targetDim = 250, doDenoiseOpening = True, doDenoiseClosing = True, closingWidth = 13, closingHeight = 3):

	#Resize to specified pixels max edge size
	scaling = 1.
	if im.shape[0] > im.shape[1]:
		if im.shape[0] != targetDim:
			scaling = float(targetDim) / im.shape[0]
			im = misc.imresize(im, (targetDim, int(round(im.shape[1] * scaling))))
	else:
		if im.shape[1] != targetDim:
			scaling = float(targetDim) / im.shape[1]
			im = misc.imresize(im, (int(round(im.shape[0] * scaling)), targetDim))
	#print "scaling", scaling

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

	#print "Threshold", threshold

	binIm = diff > threshold
	misc.imsave("threshold.png", binIm)
	#print vals.shape
	#plt.plot(vals)
	#plt.show()

	#Denoise
	diamond = morph.diamond(2)
	if doDenoiseOpening:
		currentIm = morph.binary_opening(binIm, diamond)
	else:
		currentIm = binIm
	if doDenoiseClosing:
		denoiseIm2 = morph.binary_closing(currentIm, np.ones((closingHeight, closingWidth)))
	else:
		denoiseIm2 = currentIm

	#print "currentIm", currentIm.min(), currentIm.max(), currentIm.mean()
	#print "denoiseIm2", denoiseIm2.min(), denoiseIm2.max(), currentIm.mean()
	misc.imsave("denoised1.png", currentIm * 255)
	misc.imsave("denoised2.png", denoiseIm2 * 255)

	#Number candidate regions
	#print "Numbering regions"
	numberedRegions, maxRegionNum = morph.label(denoiseIm2, 4, 0, return_num = True)
	return numberedRegions, scaling

if __name__ == "__main__":

	fina = None
	if len(sys.argv) >= 2:
		fina = sys.argv[1]
	if fina is None:
		print "Specify input image on command line"
		exit(0)

	im = misc.imread(fina)

	numberedRegions, scaling = ProcessImage(im, 250, False, False, 5, 3)

	if not os.path.exists("candidates"):
		os.mkdir("candidates")

	scores1 = ScoreUsingAspect(numberedRegions, vis="firstcritera.png")
	print "Using first criteria", scores1[0]

	for i, can in enumerate(scores1):
		bbox = can[2]
		patchIm = ExtractPatch(im, [bbox[0][0] / scaling, (bbox[0][1]+1) / scaling, bbox[1][0] / scaling, (bbox[1][1]+1)/scaling])
		#patchIm = im[bbox[1][0]:bbox[1][1],:]
		#patchIm = patchIm[:,bbox[0][0]:bbox[0][1]]
		scaledBBox = [(c[0] / scaling, c[1] / scaling) for c in bbox]
		misc.imsave("candidates/1-{0}.png".format(i), patchIm)

		outRecord = can[:]
		outRecord[2] = scaledBBox
		pickle.dump(outRecord, open("candidates/1-{0}.dat".format(i), "wb"), protocol=-1)

	scores2 = ScoreUsingSize(numberedRegions, vis="secondcriteria.png")
	print "Using second criteria", scores2[0]

	for i, can in enumerate(scores2):
		bbox = can[2]
		patchIm = ExtractPatch(im, [bbox[0][0] /scaling, (bbox[0][1]+1)/scaling, bbox[1][0]/scaling, (bbox[1][1]+1)/scaling])
		scaledBBox = [(c[0] / scaling, c[1] / scaling) for c in bbox]
		misc.imsave("candidates/2-{0}.png".format(i), patchIm)

		outRecord = can[:]
		outRecord[2] = scaledBBox
		pickle.dump(outRecord, open("candidates/2-{0}.dat".format(i), "wb"), protocol=-1)

