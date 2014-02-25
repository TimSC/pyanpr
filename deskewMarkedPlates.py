
import readannotation, deskew
import scipy.misc as misc

if __name__=="__main__":
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0

	for photo in plates:
		fina = photo[0]['file']
		im = misc.imread(fina)

		for plate in photo[1:]:
			bbox = plate['bbox']
			reg = plate['reg']
			
			xran = (bbox[0], bbox[0]+bbox[2])
			yran = (bbox[1], bbox[1]+bbox[3])

			print fina, xran, yran
			bbox, bestInd, bestAngle, rotIm = deskew.Deskew(im, (xran, yran))

			misc.imsave("rotIm{0}.png".format(count), rotIm)			
			count += 1

