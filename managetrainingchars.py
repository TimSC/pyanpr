
import readannotation, os, pickle
import scipy.misc as misc
import deskew, deskewMarkedPlates, detectblobs

if __name__=="__main__":
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0

	for photo in plates:
		fina = photo[0]['file']
		im = misc.imread(fina)

		finaImSplitPath = os.path.split(fina)
		finaImSplitExt = os.path.splitext(finaImSplitPath[1])
		finaDeskew1 = "train/" + finaImSplitExt[0] +".deskew"
		finaDeskew2 = finaImSplitPath[0] + "/" +finaImSplitExt[0] +".deskew"

		bbox, angle = None, None
		if os.path.isfile(finaDeskew1):
			bbox, angle = pickle.load(open(finaDeskew1))
		if bbox is None and os.path.isfile(finaDeskew2):
			bbox, angle = pickle.load(open(finaDeskew2))
	
		if bbox is None:
			print "Cannot find deskew file for", fina
			continue

		rotIm = deskew.RotateAndCrop(im, bbox, angle)	
		scoreIm = deskewMarkedPlates.RgbToPlateBackgroundScore(rotIm)
		charBboxes = detectblobs.DetectCharacters(scoreIm)

		print len(charBboxes)

