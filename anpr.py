
import scipy.misc as misc
import scipy.signal as signal
import numpy as np
import skimage.morphology as morph
import matplotlib.pyplot as plt

if __name__ == "__main__":

	#Improved number plate localisation algorithm and its efficient 
	#field programmable gate arrays implementation
	#Xiaojun Zhai, Faycal Bensaali, Soodamani Ramalingam

	im = misc.imread("../pyanpr-data/56897161_d613d63bce_b.jpg")
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]

	#Highlight number plate
	imnorm = greyim / 255.
	se = np.ones((3, 30))
	opim = morph.opening(imnorm, se)

	diff = greyim - opim + 128.

	#Binarize image
	vals = diff.copy()
	vals = vals.reshape((vals.size, 1))
	vals.sort()
	print vals.size, vals.shape
	threshold = vals[vals.size * 0.98]
	print threshold

	#plt.hist(vals, bins=100)
	#plt.show()

	binIm = diff > threshold


	misc.imsave("opim.png", binIm)

	

