
import interestpoints, sys, pickle
import scipy.misc as misc
import skimage.exposure as exposure
import skimage.color as color
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

	im = misc.imread(sys.argv[1])
	normIm = exposure.rescale_intensity(im)
	hsvImg = color.rgb2hsv(normIm)
	print hsvImg.shape

	featGrid, overlapGrid = interestpoints.GenerateFeatureGrid(hsvImg)

	regressor, scalingGroups = pickle.load(open("localise-model.dat", "rb"))

	scoreImg = []

	for i, featCol1 in enumerate(featGrid):

		overlapCol = overlapGrid[i]
		scoreCol = []

		for j, feats1 in enumerate(featCol1):

			whitenedFeatures = feats1[0] / scalingGroups[0]
			#print whitenedFeatures

			pred = regressor.predict(whitenedFeatures)[0]
			print i, j, pred
			
			
			scoreCol.append(pred)

		scoreImg.append(scoreCol)
	
	scoreImg = np.array(scoreImg)
	print scoreImg.shape
	scoreImg = exposure.rescale_intensity(scoreImg)
	plt.imshow(scoreImg)
	plt.show()

