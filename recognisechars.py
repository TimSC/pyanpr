import readannotation, os, pickle, random, cStringIO, trainchars, managetrainingchars
import scipy.misc as misc
import deskew, deskewMarkedPlates, detectblobs
import numpy as np
import skimage.exposure as exposure
from PIL import Image

#Average character colleration of intensity			0.670

def CompareExampleToTraining(bwImg, preProcessedModel):

	charScores = []
	for ch in preProcessedModel:
		
		examples = preProcessedModel[ch]
		scores = []
		for example in examples:
			flatExample = example.reshape(example.size)
			flatBwImg = bwImg.reshape(bwImg.size)
			score = np.corrcoef(flatExample, flatBwImg)[0,1]
			scores.append(score)
		scores = np.array(scores)
		#print "Compare to", ch, scores
		charScores.append((scores.mean(), ch))

	charScores.sort(reverse=True)
	for score, ch in charScores[:5]:
		print ch, score
	return charScores

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

	print "Preprocessing training data"
	#Preprocess training data
	preProcessedModel = {}
	for char in model:
		#print char
		procChar = []
		for example in model[char]:
			img = Image.open(cStringIO.StringIO(example))
			imgArr = np.array(img)
			imgArr = exposure.rescale_intensity(imgArr)
			bwImg = deskewMarkedPlates.RgbToPlateBackgroundScore(imgArr)
			#print bwImg.shape
			procChar.append(bwImg)
		preProcessedModel[char] = procChar
	print "Preprocessing done"

	#Iterate over test plates
	hit, miss = 0, 0
	for plateCount, objId in enumerate(testObjIds):
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
		
		print "Plate", plateCount, "of", len(testObjIds)

		for i, (bbx, cCofG) in enumerate(zip(bboxes, charCofG)):
			expectedChar = None
			if plateStrStrip is not None and len(plateStrStrip) == len(bboxes):
				expectedChar = plateStrStrip[i]

			print reg, expectedChar, cCofG, bbx
			originalHeight = bbx[3] - bbx[2]
			scaling = 50. / originalHeight
			targetMargin = 40
			margin = targetMargin / scaling

			patch = trainchars.ExtractPatch(rotIm, (cCofG[1]-margin, cCofG[1]+margin, cCofG[0]-margin, cCofG[0]+margin))
			
			#Scale height
			#height = int(round(patch.shape[0]*scaling))
			#width = int(round(patch.shape[1]*scaling))
			resizedPatch = misc.imresize(patch, (2*targetMargin, 
				2*targetMargin, patch.shape[2]))

			normIm = exposure.rescale_intensity(resizedPatch)
			bwImg = deskewMarkedPlates.RgbToPlateBackgroundScore(normIm)

			#Compare to stored examples
			charScores = CompareExampleToTraining(bwImg, preProcessedModel)

			expectedCharFiltered = expectedChar
			bestCharFiltered = charScores[0][1]
			if expectedCharFiltered == "I": expectedCharFiltered = "1"
			if expectedCharFiltered == "O": expectedCharFiltered = "0"
			if bestCharFiltered == "I": bestCharFiltered = "1"
			if bestCharFiltered == "O": bestCharFiltered = "0"

			if expectedChar is not None:
				if charScores[0][1] == expectedChar:
					hit += 1
				else:
					miss += 1

		print "Plate", plateCount, "of", len(testObjIds)
		print "Hits: {0} ({1})\tMisses: {2} ({3})".format(hit, float(hit)/(hit+miss), miss, float(miss)/(hit+miss))


