import readannotation, os, pickle, random, cStringIO, trainchars, managetrainingchars
import scipy.misc as misc
import deskew, deskewMarkedPlates, detectblobs
import numpy as np
import skimage.exposure as exposure
from PIL import Image

def CompareExampleToTraining(bwImg, preProcessedModel):
	pass

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
	trainObjIds, testObjIds, model = pickle.load(open("charmodel.dat", "rb"))
	print "Loading done"

	#Preprocess training data
	preProcessedModel = {}
	for char in model:
		print char
		procChar = []
		for example in model[char]:
			img = Image.open(cStringIO.StringIO(example))
			imgArr = np.array(img)
			imgArr = exposure.rescale_intensity(imgArr)
			bwImg = deskewMarkedPlates.RgbToPlateBackgroundScore(imgArr)
			print bwImg.shape
			procChar.append(bwImg)
		preProcessedModel[char] = procChar

	#Iterate over test plates
	for objId in testObjIds:
		for photoNum, photo in enumerate(plates):
			fina = photo[0]['file']
			reg = photo[1]['reg']
			foundObjId = photo[1]['object']
			if foundObjId == objId:
				break

		bboxes = plateCharBboxes[objId]
		plateStr = plateString[objId]
		charCofG = plateCharCofGs[objId]
		bbox, angle	= plateCharBboxAndAngle[objId]
		plateStrStrip = managetrainingchars.StripInternalSpaces(plateStr)

		#ViewPlate(fina, bbox, angle, bboxes, charCofG)
		im = misc.imread(fina)
		rotIm = deskew.RotateAndCrop(im, bbox, angle)
		
		for bbx, cCofG in zip(bboxes, charCofG):
			print reg, cCofG, bbx
			originalHeight = bbx[3] - bbx[2]
			scaling = 50. / originalHeight
			margin = 40. / scaling

			patch = trainchars.ExtractPatch(rotIm, (cCofG[1]-margin, cCofG[1]+margin, cCofG[0]-margin, cCofG[0]+margin))
			
			#Scale height
			resizedPatch = misc.imresize(patch, (int(round(patch.shape[0]*scaling)), 
				int(round(patch.shape[1]*scaling)), patch.shape[2]))

			normIm = exposure.rescale_intensity(resizedPatch)
			bwImg = deskewMarkedPlates.RgbToPlateBackgroundScore(normIm)

			#Compare to stored examples
			CompareExampleToTraining(bwImg, preProcessedModel)

