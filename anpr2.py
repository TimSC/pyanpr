import pickle, sys
import scipy.misc as misc
from skimage.transform import hough_line, hough_line_peaks

if __name__=="__main__":

	finaIm = None
	finaDat = None

	if len(sys.argv) >= 2:
		finaIm = sys.argv[1]
	if len(sys.argv) >= 3:
		finaDat = sys.argv[2]

	if finaIm is None or finaDat is None:
		print "Specify input image and data on command line (2 args)"
		exit(0)
	im = misc.imread(finaIm)

	roi = pickle.load(open(finaDat,"rb"))
	bbox = roi[2]
	print bbox

	im = im[bbox[1][0]:bbox[1][1],:]
	im = im[:,bbox[0][0]:bbox[0][1]]

	im = misc.imsave("test.png", im)
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]

	h, theta, d = hough_line(greyim)
	print h, theta, d
