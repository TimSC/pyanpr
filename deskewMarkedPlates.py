
import readannotation, deskew
import scipy.misc as misc
import skimage.color as color
import numpy as np
import math

def RgbToPlateBackgroundScore(im):
	#This function compares images to find number plate background colour
	#since UK plates come in two different colours, they are merged into
	#a single score here.

	hsvImg = color.rgb2hsv(rotIm)

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

	for photo in plates:
		fina = photo[0]['file']
		im = misc.imread(fina)

		for plate in photo[1:]:
			bbox = plate['bbox']
			reg = plate['reg']
			
			xran = (bbox[0], bbox[0]+bbox[2])
			yran = (bbox[1], bbox[1]+bbox[3])

			print fina, xran, yran
			bbox, bestInd, bestAngle = deskew.Deskew(im, (xran, yran))
			rotIm = deskew.RotateAndCrop(im, (xran, yran), bestAngle)

			#misc.imsave("rotIm{0}.png".format(count), rotIm)			

			imScore = RgbToPlateBackgroundScore(rotIm)

			misc.imsave("rotIm{0}.png".format(count), imScore)
			
			if 0:
				hsvImg = color.rgb2hsv(rotIm)

				#Remove hue information
				fixH = hsvImg.copy()
				fixH[:,:,0] = np.zeros((fixH.shape[0], fixH.shape[1]))
				fixHrgb = color.hsv2rgb(fixH)

				misc.imsave("fixH{0}.png".format(count), fixHrgb)

				#Remove saturation information
				fixS = hsvImg.copy()
				fixS[:,:,1] = np.ones((fixS.shape[0], fixS.shape[1])) * 128.
				fixSrgb = color.hsv2rgb(fixS)

				misc.imsave("fixS{0}.png".format(count), fixSrgb)

				#Remove intensity information
				fixV = hsvImg.copy()
				fixV[:,:,2] = np.ones((fixV.shape[0], fixV.shape[1])) * 128.
				fixVrgb = color.hsv2rgb(fixV)

				misc.imsave("fixV{0}.png".format(count), fixVrgb)


			count += 1

