
import cv2
import scipy.misc as misc
import readannotation, sys, os, localiseplate
import numpy as np
import skimage.exposure as exposure

def test():
	im1 = cv2.imread("/media/data/home/tim/kinatomic/datasets/anpr-plates/IMG_20140219_105833.jpg")

	im1 = misc.imresize(im1, (800, 600))

	print "Convert to grey"
	grey1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	print "Conversion done"

	print "GetKeypoints"
	detector = cv2.FeatureDetector_create("BRISK")
	#print str(detector.getParams())
	#detector.setInt("nFeatures", 50)
	print "GetKeypoints done"

	print "Get descriptors"
	descriptor = cv2.DescriptorExtractor_create("BRISK")
	#print "Extracting points of interest 1"
	keypoints1 = detector.detect(grey1)
	#keypoints1 = DetectAcrossImage(grey1, detector)
	#VisualiseKeypoints(grey1, keypoints1)
	(keypoints1, descriptors1) = descriptor.compute(grey1, keypoints1)
	print "Get descriptors done"

	for kp in keypoints1:
		#print kp.pt
		im1[kp.pt[1], kp.pt[0], :] = (255, 0, 0)
	
	misc.imshow(im1)


def ExtractHistogram(im, chanBins):
	for chanNum in range(im.shape[2]):
		chanIm = im[:,:,chanNum]
		bins = chanBins[chanNum]
		flatChanIm = chanIm.reshape((chanIm.size,))
		#print flatChanIm.shape, bins
		freq, bins2 = np.histogram(flatChanIm, bins, normed=1)
		print freq

def OverlapProportion(patchBbox, plateBbox):
	print patchBbox, plateBbox

if __name__ == "__main__":

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
		im = misc.imread(actualFina)
		normIm = exposure.rescale_intensity(im)

		for record in photo[1:]:
			plateBbox = record['bbox']
			#print plateBbox
			wx = 50
			wy = 50

			for cx in range(0, normIm.shape[1], 30):
				for cy in range(0, normIm.shape[1], 30):
					#print cx, cy
					x1 = cx - wx / 2.
					x2 = cx + wx / 2.
					y1 = cy - wy / 2.
					y2 = cy + wy / 2.
					patchBbox = (x1, x2, y1, y2)

					crop = localiseplate.ExtractPatch(normIm, patchBbox)
					ExtractHistogram(crop, [np.linspace(0, 255, 10), np.linspace(0, 255, 10), np.linspace(0, 255, 10)])
					#misc.imshow(crop)

					OverlapProportion(patchBbox, plateBbox)

