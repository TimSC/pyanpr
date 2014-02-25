
import readannotation, deskew
import scipy.misc as misc
import skimage.color as color
import numpy as np

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

