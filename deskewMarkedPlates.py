
import readannotation, deskew
import scipy.misc as misc
import skimage.color as color
import skimage.exposure as exposure
import numpy as np
import math, pickle, os, sys

def RgbToPlateBackgroundScore(im):
	#This function compares images to find number plate background colour
	#since UK plates come in two different colours, they are merged into
	#a single score here.

	hsvImg = color.rgb2hsv(im)

	#Target colours
	#Yellow HSV 47/360, 81/100, 100/100 or 32/255, 217/255, > 248/255
	#While HSV None, < 8/100, > 82/100 or ?/255, < 21/255, > 255/255	

	#Compare to white
	whiteScore = (255.-hsvImg[:,:,1]) * (hsvImg[:,:,2]) / pow(255., 2.)
	
	#Compare to yellow
	#Hue is a repeating value similar to angle, compute dot product with yellow
	hueAng = hsvImg[:,:,0] * (2. * math.pi / 255.)
	hueSin = np.sin(hueAng)
	hueCos = np.cos(hueAng)
	targetSin = math.sin(math.radians(47.)) #Hue of yellow
	targetCos = math.cos(math.radians(47.))
	dotProd = hueSin * targetSin + hueCos * targetCos
	yellowHueScore = (dotProd + 1.) / 2. #Scale from 0 to 1	
	yellowSatScore = np.abs(hsvImg[:,:,1] - 217.) / 217.
	yellowValScore = hsvImg[:,:,2] / 255.
	yellowScore = yellowHueScore * yellowSatScore * yellowValScore

	scoreMax = np.maximum(whiteScore, yellowScore)
	return scoreMax

if __name__=="__main__":
	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0

	imgPath = None
	if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
		imgPath = sys.argv[1]

	if not os.path.exists("train"):
		os.mkdir("train")

	for photo in plates:
		fina = photo[0]['file']

		actualFina = readannotation.GetActualImageFileName(fina, [imgPath])

		if actualFina is None:
			print "Image file not found:", fina
			continue

		im = misc.imread(actualFina)

		finaImSplitPath = os.path.split(fina)
		finaImSplitExt = os.path.splitext(finaImSplitPath[1])
		outRootFina = "train" + "/" + finaImSplitExt[0]

		for plate in photo[1:]:
			bbox = plate['bbox']
			reg = plate['reg']
			
			xran = (bbox[0], bbox[0]+bbox[2])
			yran = (bbox[1], bbox[1]+bbox[3])

			print count, actualFina, xran, yran
			bbox, bestInd, bestAngle = deskew.Deskew(im, (xran, yran))
			rotIm = deskew.RotateAndCrop(im, (xran, yran), bestAngle)

			#misc.imsave("rotIm{0}.png".format(count), rotIm)			

			imScore = RgbToPlateBackgroundScore(rotIm)

			#normContrast = exposure.equalize_hist(imScore)
			normContrast = exposure.rescale_intensity(imScore)
			#normContrast = exposure.equalize_adapthist(imScore)

			thresh = 0.6 * (normContrast.min() + normContrast.max())
			#normContrast = (normContrast > 0.5)
			#print normContrast.min(), normContrast.max()

			misc.imsave("{0}.png".format(outRootFina), normContrast)
			pickle.dump((bbox, bestAngle), open("{0}.deskew".format(outRootFina), "wb"), protocol=-1)

			if 0:
				import matplotlib.pyplot as plt
				dat = normContrast.reshape((normContrast.size,))

				plt.subplot(3,1,1)
				ims = plt.imshow(normContrast)
				ims.set_cmap('gray')

				plt.subplot(3,1,2)
				ims = plt.imshow(normContrast > thresh)
				ims.set_cmap('gray')

				plt.subplot(3,1,3)
				plt.hist(dat, bins=256)
				plt.show()

			count += 1

