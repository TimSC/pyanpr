
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
		charBboxes, charCofG = detectblobs.DetectCharacters(scoreIm)

		print len(charBboxes)
		rotImMod = rotIm.copy()

		for cg in charCofG:
			try:
				rotImMod[round(cg[0]), round(cg[1]), :] = (128, 128, 250)
			except:
				pass

		mergedChars = None
		sepImg = None

		for cb in charBboxes:
			im2 = rotImMod[cb[2]:cb[3]+1,:,:]
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

		import matplotlib.pyplot as plt
		plt.clf()
		plt.subplot(211)
		plt.imshow(rotIm)
		plt.subplot(212)
		plt.imshow(mergedChars)
		plt.savefig(finaImSplitExt[0]+".png")

