import readannotation, os, pickle
import scipy.misc as misc
import deskew, deskewMarkedPlates, detectblobs
import numpy as np

def ViewPlate(fina, bbox, angle, charBboxes, charCofG):

	print fina
	im = misc.imread(fina)
	rotIm = deskew.RotateAndCrop(im, bbox, angle)	
	scoreIm = deskewMarkedPlates.RgbToPlateBackgroundScore(rotIm)

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
	#plt.savefig(finaImSplitExt[0]+".png")
	plt.show()

def StripInternalSpaces(strIn):
	if strIn is None: return None
	out = []
	for ch in strIn:
		if ch != " ":
			out.append(ch)
	return "".join(out)

if __name__=="__main__":
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0
	print "Num photos", len(plates)
	plateCharBboxes = {}
	plateCharCofGs = {}
	plateCharBboxAndAngle = {}
	plateString = {}

	print "Loading"
	plateCharBboxes = pickle.load(open("charbboxes.dat", "r"))
	plateCharCofGs = pickle.load(open("charcofgs.dat", "r"))
	plateCharBboxAndAngle = pickle.load(open("charbboxangle.dat", "r"))
	plateString = pickle.load(open("charstrings.dat", "r"))
	print "Loading done"

	objIds = plateString.keys()
	objIds.sort()
	for objId in objIds:
		for photoNum, photo in enumerate(plates):
			fina = photo[0]['file']
			foundObjId = photo[1]['object']
			if foundObjId == objId:
				break

		bboxes = plateCharBboxes[objId]
		plateStr = plateString[objId]
		charCofG = plateCharCofGs[objId]
		bbox, angle	= plateCharBboxAndAngle[objId]

		plateStrStrip = StripInternalSpaces(plateStr)

		print photoNum, plateStrStrip, len(bboxes)
		if plateStrStrip is not None and len(plateStrStrip) != len(bboxes):
			print "Bbox number mismatch"

		ViewPlate(fina, bbox, angle, bboxes, charCofG)

