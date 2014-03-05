import readannotation, os, pickle, random, cStringIO, managetrainingchars, sys, localiseplate
import scipy.misc as misc
import deskew, deskewMarkedPlates, detectblobs
import numpy as np
import skimage.exposure as exposure
from PIL import Image

if __name__=="__main__":
	imgPath = None
	if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
		imgPath = sys.argv[1]

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

		actualFina = readannotation.GetActualImageFileName(fina, [imgPath])

		if actualFina is None:
			print "Image file not found:", fina
			continue

		bboxes = plateCharBboxes[objId]
		plateStr = plateString[objId]
		charCofG = plateCharCofGs[objId]
		bbox, angle	= plateCharBboxAndAngle[objId]

		if plateStr is None: continue #Bad plate

		plateStrStrip = managetrainingchars.StripInternalSpaces(plateStr)

		print photoNum, plateStrStrip, len(bboxes)
		if plateStrStrip is not None and len(plateStrStrip) != len(bboxes):
			print "Bbox number mismatch"

		#ViewPlate(actualFina, bbox, angle, bboxes, charCofG)
		im = misc.imread(actualFina)
		rotIm = deskew.RotateAndCrop(im, bbox, angle)
		
		for char, bbx, cCofG in zip(plateStrStrip, bboxes, charCofG):
			print char, cCofG, bbx
			originalHeight = bbx[3] - bbx[2]
			scaling = 50. / originalHeight
			print scaling
			targetMargin = 40
			margin = targetMargin / scaling

			patch = localiseplate.ExtractPatch(rotIm, (cCofG[1]-margin, cCofG[1]+margin, cCofG[0]-margin, cCofG[0]+margin))
			
			#Scale height
			#patchWidth = int(round(patch.shape[0]*scaling))
			patchWidth = targetMargin * 2

			resizedPatch = misc.imresize(patch, (targetMargin * 2, 
				patchWidth, patch.shape[2]))

			#normIm = exposure.rescale_intensity(resizedPatch)
			normIm = resizedPatch

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

			model[char].append((binImage.getvalue(), objId))

	pickle.dump((trainObjIds, testObjIds, model), open("charmodel.dat", "wb"), protocol=-1)

