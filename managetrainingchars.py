
import readannotation, os, pickle
import scipy.misc as misc
import deskew, deskewMarkedPlates, detectblobs
import numpy as np

if __name__=="__main__":
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0

	for photo in plates:
		fina = photo[0]['file']
		print fina
		im = misc.imread(fina)

		finaImSplitPath = os.path.split(fina)
		finaImSplitExt = os.path.splitext(finaImSplitPath[1])
		finaDeskew1 = "train/" + finaImSplitExt[0] +".deskew"
		finaDeskew2 = finaImSplitPath[0] + "/" +finaImSplitExt[0] +".deskew"

		bbox, angle = None, None
		if os.path.isfile(finaDeskew1):
			bbox, angle = pickle.load(open(finaDeskew1))
		if bbox is None and os.path.isfile(finaDeskew2):
			bbox, angle = pickle.load(open(finaDeskew2))
	
		if bbox is None:
			print "Cannot find deskew file for", fina
			continue

		rotIm = deskew.RotateAndCrop(im, bbox, angle)	
		scoreIm = deskewMarkedPlates.RgbToPlateBackgroundScore(rotIm)
		print "Find characters"
		charBboxes = detectblobs.DetectCharacters(scoreIm)

		print len(charBboxes)

		mergedChars = None
		sepImg = None

		for cb in charBboxes:
			im2 = rotIm[cb[2]:cb[3]+1,:,:]
			im3 = im2[:,cb[0]:cb[1]+1,:]
			if mergedChars is None:
				mergedChars = im3
				sepImg = np.ones((im3.shape[0], 10, 3)) * 128
			else:
				mergedChars = np.hstack((mergedChars, sepImg, im3))

		import matplotlib.pyplot as plt
		plt.clf()
		plt.subplot(211)
		plt.imshow(rotIm)
		plt.subplot(212)
		plt.imshow(mergedChars)
		plt.savefig(finaImSplitExt[0]+".png")

