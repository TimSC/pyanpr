
import recognisechars, readannotation, localiseplate
import deskew, deskewMarkedPlates, detectblobs
import scipy.misc as misc

def AnprLocalised(im, bbox):
	#Deskew
	bbox, bestInd, bestAngle = deskew.Deskew(im, bbox)
	deskewedIm = deskew.RotateAndCrop(im, bbox, bestAngle)

	#Split into characters
	scoreIm = deskewMarkedPlates.RgbToPlateBackgroundScore(deskewedIm)
	charBboxes, charCofGs = detectblobs.DetectCharacters(scoreIm)

	for bbx, cofg in zip(charBboxes, charCofGs):
		charScores, candidateImgs = recognisechars.ProcessPatch(scoreIm, bbx)
		print charSchores[:5]




def Anpr(im, preProcessedModel):
	#Localise
	numberedRegions = localiseplate.ProcessImage(im)

	scores1 = localiseplate.ScoreUsingAspect(numberedRegions, "firstcritera.png")
	print "Using aspect criteria", scores1[0]

	for canNum, candidate in enumerate(scores1[:10]):
		print canNum
		AnprLocalised(im, candidate[2])

	scores2 = localiseplate.ScoreUsingSize(numberedRegions, "secondcriteria.png")
	print "Using size criteria", scores2[0]

	for canNum, candidate in enumerate(scores2[:10]):
		print canNum
		AnprLocalised(im, candidate[2])

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


