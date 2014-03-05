import readannotation, os, pickle, random, cStringIO, trainchars, managetrainingchars
import scipy.misc as misc
import deskew, deskewMarkedPlates, detectblobs
import numpy as np
import skimage.exposure as exposure
from PIL import Image
import skimage.filter as filt

#80x80 Average character correlation of intensity			0.670
#80x80 Max character correlation of intensity				0.661
#80x80 Min template difference								0.557
#80x80 Mean character template difference					0.600
#40x40 Min template difference 								0.852
#40x40 Max character correlation with blur					0.916

def CompareExampleToTraining(bwImg, preProcessedModel, getCandidates = False):

	bwImg = filt.gaussian_filter(bwImg, 2.)

	bwImg = bwImg[20:-20,:]
	bwImg = bwImg[:,20:-20]

	charScores = []
	for ch in preProcessedModel:
		
		examples = preProcessedModel[ch]
		for example, sourceObjId in examples:
			#Tight crop
			example = filt.gaussian_filter(example, 2.)
			example = example[20:-20,:]
			example = example[:,20:-20]

			flatExample = example.reshape(example.size)
			flatBwImg = bwImg.reshape(bwImg.size)
			if 0:
				den = np.abs(flatExample-flatBwImg).mean()
				if den > 0.:
					score = 1. / den
				else:
					score = 100.
			if 1:
				score = np.corrcoef(flatExample, flatBwImg)[0,1]
			
			charScores.append((score, ch, example, sourceObjId))

	charScores.sort(reverse=True)
	#for score, ch, example, sourceObjId in charScores[:5]:
	#	#annot = GetAnnotForObjId(plates, sourceObjId)

	#	print ch, score, sourceObjId#, annot['reg']

	mergeImg = None
	if getCandidates:
		mergeImg = bwImg.copy()
		for score, ch, example, sourceObjId in charScores[:10]:
			mergeImg = np.hstack((mergeImg, example))
		#misc.imshow(mergeImg)

	return charScores, mergeImg

def GetPhotoForObjId(plates, objId):
	for photo in plates:
		for annot in photo[1:]:
			if annot['object'] == objId:
				return photo
	return None

def GetAnnotForObjId(plates, objId):
	for photo in plates:
		for annot in photo[1:]:
			if annot['object'] == objId:
				return annot
	return None

def LoadModel():
	plateCharBboxes = pickle.load(open("charbboxes.dat", "r"))
	plateCharCofGs = pickle.load(open("charcofgs.dat", "r"))
	plateCharBboxAndAngle = pickle.load(open("charbboxangle.dat", "r"))
	plateString = pickle.load(open("charstrings.dat", "r"))
	trainObjIds, testObjIds, model = pickle.load(open("charmodel.dat", "rb"))
	return plateCharBboxes, plateCharCofGs, plateCharBboxAndAngle, \
		plateString, trainObjIds, testObjIds, model

def PreprocessTraining(model):

	if len(model)==0:
		raise RuntimeError("Recognition model is empty")

	#Preprocess training data
	preProcessedModel = {}

	for char in model:
		#print char
		procChar = []
		for example, sourceObjId in model[char]:
			img = Image.open(cStringIO.StringIO(example))
			imgArr = np.array(img)
			imgArr = exposure.rescale_intensity(imgArr)
			bwImg = deskewMarkedPlates.RgbToPlateBackgroundScore(imgArr)
			#print bwImg.shape
			procChar.append((bwImg, sourceObjId))
		preProcessedModel[char] = procChar
	return preProcessedModel

def ProcessPatch(rotIm, bbx, cCofG, preProcessedModel):

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
	charScores, candidateImgs = CompareExampleToTraining(bwImg, preProcessedModel, True)
	return charScores, candidateImgs

if __name__=="__main__":
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0
	print "Num photos", len(plates)

	print "Loading and Preprocessing training data"
	plateCharBboxes, plateCharCofGs, plateCharBboxAndAngle, \
		plateString, trainObjIds, testObjIds, model = LoadModel()

	preProcessedModel = PreprocessTraining(model)
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
			charScores, candidateImgs = ProcessPatch(rotIm, bbx, cCofG, preProcessedModel)

			expectedCharFiltered = expectedChar
			bestCharFiltered = charScores[0][1]
			if expectedCharFiltered == "I": expectedCharFiltered = "1"
			if expectedCharFiltered == "O": expectedCharFiltered = "0"
			if bestCharFiltered == "I": bestCharFiltered = "1"
			if bestCharFiltered == "O": bestCharFiltered = "0"

			if expectedChar is not None:
				if bestCharFiltered == expectedCharFiltered:
					hit += 1
				else:
					#misc.imshow(candidateImgs)
					misc.imsave("miss{0}.png".format(miss), candidateImgs)
					missFi = open("miss{0}.txt".format(miss), "wt")
					missFi.write("{0},{1},{2},{3}\n".format(reg, expectedChar, cCofG, bbx))
					for sc in charScores[:5]:
						annot = GetAnnotForObjId(plates, sc[3])
						missFi.write("{0},{1}\n".format(sc, annot['reg']))
					miss += 1

		print "Plate", plateCount, "of", len(testObjIds)
		print "Hits: {0} ({1})\tMisses: {2} ({3})".format(hit, float(hit)/(hit+miss), miss, float(miss)/(hit+miss))


