
import recognisechars, readannotation, localiseplate
import deskew, deskewMarkedPlates, detectblobs
import scipy.misc as misc

def AnprLocalised(im, bbox, preProcessedModel):
	#Deskew
	expandedBbox, bestInd, bestAngle = deskew.Deskew(im, bbox)
	#print "expandedBbox", expandedBbox
	deskewedIm = deskew.RotateAndCrop(im, expandedBbox, bestAngle)

	#Split into characters
	scoreIm = deskewMarkedPlates.RgbToPlateBackgroundScore(deskewedIm)
	#misc.imshow(scoreIm)

	charBboxes, charCofGs = detectblobs.DetectCharacters(scoreIm)

	if charBboxes == None or charCofGs == None:
		return None

	#Recognise individual characters
	details = []
	bestGuess = ""
	for bbx, cofg in zip(charBboxes, charCofGs):
		charScores, candidateImgs = recognisechars.ProcessPatch(deskewedIm, bbx, cofg, preProcessedModel)
		#for ch in charScores[:5]:
		#	print ch[0], ch[1]
		details.append(charScores)
		bestGuess += charScores[0][1]
	return bestGuess, details

def Anpr(im, preProcessedModel):
	#Localise
	numberedRegions, scaling = localiseplate.ProcessImage(im)

	scores1 = localiseplate.ScoreUsingAspect(numberedRegions, "firstcritera.png")
	print "Using aspect criteria", scores1[0]

	for canNum, candidate in enumerate(scores1[:10]):
		print canNum
		plateBbox = candidate[2]
		scaledBBox = [(c[0] / scaling, c[1] / scaling) for c in plateBbox]
		bestGuess, details = AnprLocalised(im, scaledBBox, preProcessedModel)
		print bestGuess

	scores2 = localiseplate.ScoreUsingSize(numberedRegions, "secondcriteria.png")
	print "Using size criteria", scores2[0]

	for canNum, candidate in enumerate(scores2[:10]):
		print canNum
		plateBbox = candidate[2]
		scaledBBox = [(c[0] / scaling, c[1] / scaling) for c in plateBbox]
		bestGuess, details = AnprLocalised(im, scaledBBox, preProcessedModel)
		print bestGuess

if __name__=="__main__":
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0
	print "Num photos", len(plates)

	print "Loading and Preprocessing training data"
	plateCharBboxes, plateCharCofGs, plateCharBboxAndAngle, \
		plateString, trainObjIds, testObjIds, model = recognisechars.LoadModel()

	preProcessedModel = recognisechars.PreprocessTraining(model)
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

			im = misc.imread(fina)
		
			Anpr(im, preProcessedModel)


