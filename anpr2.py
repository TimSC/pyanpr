import pickle, sys, math
import scipy.misc as misc
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
import numpy as np

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

	misc.imsave("test.png", im)
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]

	h, theta, d = hough_line(greyim)

	if 0:
		plt.imshow(np.log(1 + h),
		       extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
		               d[-1], d[0]],
		       cmap=plt.cm.gray, aspect=1/1.5)
		plt.title('Hough transform')
		plt.xlabel('Angles (degrees)')
		plt.ylabel('Distance (pixels)')
		plt.show()

	peaks = hough_line_peaks(h, theta, d)
	for _, angle, dist in zip(*peaks):
		print "peak", angle, math.degrees(angle), dist

	bestAngle = peaks[1][0]
	while bestAngle > math.pi / 4.:
		bestAngle -= math.pi / 2.
	while bestAngle < -math.pi / 4.:
		bestAngle += math.pi / 2.

	pickle.dump((bbox, bestAngle), open("out.deskew", "wb"), protocol=-1)


