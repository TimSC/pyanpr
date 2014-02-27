
import readannotation, os, pickle
import scipy.misc as misc
import deskew, deskewMarkedPlates, detectblobs
import numpy as np

def GetDeskewForImageFilename(fina):
	finaImSplitPath = os.path.split(fina)
	finaImSplitExt = os.path.splitext(finaImSplitPath[1])
	finaDeskew1 = "train/" + finaImSplitExt[0] +".deskew"
	finaDeskew2 = finaImSplitPath[0] + "/" +finaImSplitExt[0] +".deskew"

	bbox, angle = None, None
	if os.path.isfile(finaDeskew1):
		bbox, angle = pickle.load(open(finaDeskew1))
	if bbox is None and os.path.isfile(finaDeskew2):
		bbox, angle = pickle.load(open(finaDeskew2))
	return bbox, angle	

def SplitCharacters(fina, bbox, angle):
	
	print fina
	im = misc.imread(fina)

	rotIm = deskew.RotateAndCrop(im, bbox, angle)	
	scoreIm = deskewMarkedPlates.RgbToPlateBackgroundScore(rotIm)
	print "Find characters"
	charBboxes, charCofG = detectblobs.DetectCharacters(scoreIm)

	return charBboxes, charCofG

def Stuff():

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

if __name__=="__main__":
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0
	print "Num photos", len(plates)
	plateCharBboxes = {}
	plateCharCofGs = {}
	plateCharBboxAndAngle = {}

	while 1:
		print "1. Split characters"
		print "s. Save"
		print "q. Quit"

		userInput = raw_input(">")

		if userInput == "1":

			startIndex = raw_input("start index:")
			endIndex = raw_input("end index:")

			try:
				startIndexVal = int(startIndex)
				if startIndexVal < 0:
					startIndexVal = 0
			except:
				startIndexVal = None

			if startIndex == "":
				startIndexVal = 0

			try:
				endIndexVal = int(endIndex)
				if endIndexVal < 0:
					endIndexVal = 0
			except:
				endIndexVal = None
		
			if endIndex == "":
				endIndexVal = len(plates)
		
			if startIndexVal is not None and endIndexVal is not None:

				for photoNum in range(startIndexVal,endIndexVal):

					photo = plates[photoNum]
					fina = photo[0]['file']
					objId = photo[1]['object']

					bbox, angle	= GetDeskewForImageFilename(fina)
					if bbox is None:
						print "Cannot find deskew file for", fina
						continue

					charBboxes, charCofG = SplitCharacters(fina, bbox, angle)
					plateCharBboxes[objId] = charBboxes
					plateCharCofGs[objId] = charCofG
					plateCharBboxAndAngle[objId] = (bbox, angle)

		if userInput == "s":
			print "Saving"
			pickle.dump(plateCharBboxes, open("charbboxes.dat", "w"))
			pickle.dump(plateCharCofGs, open("charcofgs.dat", "w"))
			pickle.dump(plateCharBboxAndAngle, open("charbboxangle.dat", "w"))
			print "Saving done"

		if userInput == "q":
			print "All done!"
			exit(0)


