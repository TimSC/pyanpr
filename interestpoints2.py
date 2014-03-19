
import interestpoints, sys, pickle
import scipy.misc as misc
import skimage.exposure as exposure
import skimage.color as color

if __name__ == "__main__":

	im = misc.imread(sys.argv[1])
	normIm = exposure.rescale_intensity(im)
	hsvImg = color.rgb2hsv(normIm)

	featGrid, overlapGrid = interestpoints.GenerateFeatureGrid(hsvImg)

	regressor = pickle.load(open("localise-model.dat", "rb"))

	print regressor
