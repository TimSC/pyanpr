
import recognisechars, readannotation, localiseplate
import deskew, deskewMarkedPlates, detectblobs
import scipy.misc as misc
import sys, os

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
	bestGuessConfidence = []
	for bbx, cofg in zip(charBboxes, charCofGs):
		charScores, candidateImgs = recognisechars.ProcessPatch(deskewedIm, bbx, cofg, preProcessedModel)
		#for ch in charScores[:5]:
		#	print ch[0], ch[1]
		details.append(charScores)
		bestChar = charScores[0][1]
		bestGuess += bestChar
		
		#Look at the ranked results and see the proportion that
		#are in agreement in the top 10
		countAgree = 0
		countTested = 0
		for sc, ch, im, srcId in charScores[1:11]:
			if bestChar == ch:
				countAgree += 1
			countTested += 1

		bestGuessConfidence.append(float(countAgree) / countTested)
	return bestGuess, bestGuessConfidence, details

def Anpr(im, preProcessedModel):
	#Localise
	numberedRegions, scaling = localiseplate.ProcessImage(im)

	scores1 = localiseplate.ScoreUsingAspect(numberedRegions, "firstcritera.png")
	#print "Using aspect criteria", scores1[0]
	bestGuess = None
	bestGuessConf = []

	for canNum, candidate in enumerate(scores1[:10]):
		#print canNum
		plateBbox = candidate[2]
		scaledBBox = [(c[0] / scaling, c[1] / scaling) for c in plateBbox]
		anprRet = AnprLocalised(im, scaledBBox, preProcessedModel)
		if anprRet is None:
			continue
		guess, guessConfidence, details = anprRet	

		print canNum, guess, guessConfidence

		if sum(guessConfidence) > sum(bestGuessConf):
			bestGuess = guess
			bestGuessConf = guessConfidence

	scores2 = localiseplate.ScoreUsingSize(numberedRegions, "secondcriteria.png")
	#print "Using size criteria", scores2[0]

	for canNum, candidate in enumerate(scores2[:10]):
		#print canNum
		plateBbox = candidate[2]
		scaledBBox = [(c[0] / scaling, c[1] / scaling) for c in plateBbox]
		anprRet = AnprLocalised(im, scaledBBox, preProcessedModel)
		if anprRet is None:
			continue
		guess, guessConfidence, details = anprRet
		print canNum, guess, guessConfidence

		if sum(guessConfidence) > sum(bestGuessConf):
			bestGuess = guess
			bestGuessConf = guessConfidence

	return bestGuess, bestGuessConf

def TestOnUnseenSamples(preProcessedModel, imgPath = None):

	#Run on unseen test examples
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0
	print "Num photos", len(plates)

	#Iterate over test plates
	hit, miss = 0, 0
	for plateCount, objId in enumerate(testObjIds):
		for photoNum, photo in enumerate(plates):
			fina = photo[0]['file']
			reg = photo[1]['reg']
			foundObjId = photo[1]['object']
			if foundObjId == objId:
				break

			print fina, reg
			actualFina = readannotation.GetActualImageFileName(fina, [imgPath])
					
			if actualFina is None:
				print "Image file not found:", fina
				continue

			im = misc.imread(actualFina)
	
			guess = Anpr(im, preProcessedModel)
			print "Final plate", guess

if __name__=="__main__":

	print "Loading and Preprocessing training data"
	plateCharBboxes, plateCharCofGs, plateCharBboxAndAngle, \
		plateString, trainObjIds, testObjIds, model = recognisechars.LoadModel()

	preProcessedModel = recognisechars.PreprocessTraining(model)
	print "Preprocessing done"
	imgPath = None
	if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
		imgPath = sys.argv[1]

	if len(sys.argv) < 2 or imgPath is not None:
		TestOnUnseenSamples(preProcessedModel, imgPath)
	else:
		#Test on specified example
		im = misc.imread(sys.argv[1])

		guess = Anpr(im, preProcessedModel)
		print "Final plate", guess
		


