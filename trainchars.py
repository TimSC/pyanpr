import readannotation, os, pickle, random, cStringIO
import scipy.misc as misc
import deskew, deskewMarkedPlates, detectblobs
import numpy as np
import skimage.exposure as exposure
from PIL import Image

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

if __name__=="__main__":
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0
	print "Num photos", len(plates)
	plateCharBboxes = {}
	plateCharCofGs = {}
	plateCharBboxAndAngle = {}
	plateString = {}
	model = {}

	print "Loading"
	plateCharBboxes = pickle.load(open("charbboxes.dat", "r"))
	plateCharCofGs = pickle.load(open("charcofgs.dat", "r"))
	plateCharBboxAndAngle = pickle.load(open("charbboxangle.dat", "r"))
	plateString = pickle.load(open("charstrings.dat", "r"))
	print "Loading done"

	objIds = plateString.keys()
	objIds.sort()

	#Use half for training
	random.shuffle(objIds)
	splitInd = int(round(len(objIds) / 2.))
	trainObjIds = objIds[:splitInd]
	testObjIds = objIds[splitInd:]
	print "Total plates:", len(objIds)
	print "Training with",len(trainObjIds),"plates"
	print "Test with",len(testObjIds),"plates"

	for objId in trainObjIds:
		for photoNum, photo in enumerate(plates):
			fina = photo[0]['file']
			foundObjId = photo[1]['object']
			if foundObjId == objId:
				break

		if objId not in plateString: continue #Plate not checked

		bboxes = plateCharBboxes[objId]
		plateStr = plateString[objId]
		charCofG = plateCharCofGs[objId]
		bbox, angle	= plateCharBboxAndAngle[objId]

		if plateStr is None: continue #Bad plate

		plateStrStrip = StripInternalSpaces(plateStr)

		print photoNum, plateStrStrip, len(bboxes)
		if plateStrStrip is not None and len(plateStrStrip) != len(bboxes):
			print "Bbox number mismatch"

		#ViewPlate(fina, bbox, angle, bboxes, charCofG)
		im = misc.imread(fina)
		rotIm = deskew.RotateAndCrop(im, bbox, angle)
		
		for char, bbx, cCofG in zip(plateStrStrip, bboxes, charCofG):
			print char, cCofG, bbx
			originalHeight = bbx[3] - bbx[2]
			scaling = 50. / originalHeight
			print scaling
			margin = 40. / scaling

			patch = ExtractPatch(rotIm, (cCofG[1]-margin, cCofG[1]+margin, cCofG[0]-margin, cCofG[0]+margin))
			
			#Scale height
			resizedPatch = misc.imresize(patch, (int(round(patch.shape[0]*scaling)), 
				int(round(patch.shape[1]*scaling)), patch.shape[2]))

			normIm = exposure.rescale_intensity(resizedPatch)

			print normIm.shape
			#misc.imshow(normIm)

			if char not in model:
				model[char] = []
			pilImg = misc.toimage(resizedPatch)
			binImage = cStringIO.StringIO()
			pilImg.save(binImage, "png")

			#Merge I into 1 and O into 0
			modChar = char
			if modChar == "I": modChar = "1"
			if modChar == "O": modChar = "0"

			model[char].append(binImage.getvalue())

	pickle.dump((trainObjIds, testObjIds, model), open("charmodel.dat", "wb"), protocol=-1)

