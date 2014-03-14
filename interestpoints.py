
import cv2, pickle, random
import scipy.misc as misc
import readannotation, sys, os, localiseplate
import numpy as np
import skimage.exposure as exposure
import skimage.color as color
import sklearn.svm as svm
import sklearn.ensemble as ensemble
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

#HSV 10-bin histogram
#Average error 0.038
#Correlation 0.60
#Non-zero label error 0.44

#HSV 10-bin histogram and 20 bin 1D sobel
#Average error 0.038
#Correlation 0.62
#Non-zero label error 0.44

#20 bin 1D sobel
#Average error 0.0504484715633
#Correlation 0.398753870431
#Non-zero label error 0.530846231923

#15 bin 2D absolute sobel
#Average error 0.0477460861213
#Correlation 0.443438032678
#Non-zero label error 0.489148526301

#50 bin 2D absolute sobel
#Average error 0.0464841858963
#Correlation 0.516380023656
#Non-zero label error 0.477589826873

#100 bin 2D absolute sobel
#Average error 0.0464569322944
#Correlation 0.515796832502
#Non-zero label error 0.459122956116

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
	out = []
	for chanNum in range(im.shape[2]):
		chanIm = im[:,:,chanNum]
		bins = chanBins[chanNum]
		flatChanIm = chanIm.reshape((chanIm.size,))
		#print flatChanIm.shape, bins
		freq, bins2 = np.histogram(flatChanIm, bins, density=1)
		out.extend(freq / len(bins2))
		#print freq
	return out

def ExtractHistogramGrey(im, bins):
	flatChanIm = im.reshape((im.size,))
	#print flatChanIm.shape, bins
	freq, bins2 = np.histogram(flatChanIm, bins, density=1)
	return freq / len(bins2)

def SobelHistogram():
	pass


def OverlapProportion1D(a1, a2, b1, b2):
	if a2 <= b1: return 0.
	if a1 >= b2: return 0.
	if a1 >= b1 and a2 <= b2: return 1.
	if b1 >= a1 and b2 <= a2: 
		return float(b2 - b1) / (a2 - a1)
	if a1 >= b1 and a1 <= b2 and a2 > b2:
		return float(b2 - a1) / (a2 - a1)
	if a2 >= b1 and a2 <= b2 and a1 < b1:
		return float(a2 - b1) / (a2 - a1)
	raise RuntimeError("Internal error")

def GenerateSamples(plates, imgPath, maxZeroSamples = 500):

	labels = []
	samples = []
	plateIds = []

	for photoNum, photo in enumerate(plates):
		fina = photo[0]['file']
		print photoNum, len(plates), fina

		labelsNonZero = []
		samplesNonZero = []
		plateIdNonZero = []
		labelsZero = []
		samplesZero = []
		plateIdZero = []

		actualFina = readannotation.GetActualImageFileName(fina, [imgPath])	
		im = misc.imread(actualFina)
		normIm = exposure.rescale_intensity(im)
		hsvImg = color.rgb2hsv(normIm)

		greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]
		edgeIm1 = np.abs(ndimage.sobel(greyim, axis=0))
		edgeIm2 = np.abs(ndimage.sobel(greyim, axis=1))
		edgeImCombined = edgeIm1 + edgeIm2

		for record in photo[1:]:
			plateId = record['object']
			plateBbox = record['bbox']
			plateBbox = [plateBbox[0], plateBbox[0] + plateBbox[2], plateBbox[1], plateBbox[1]+plateBbox[3]]

			wx = 50.
			wy = 50.
			
			for cx in range(0, normIm.shape[1], 60):
				for cy in range(0, normIm.shape[0], 60):
					

					#print cx, cy
					x1 = cx - wx / 2.
					x2 = cx + wx / 2.
					y1 = cy - wy / 2.
					y2 = cy + wy / 2.
					patchBbox = (x1, x2, y1, y2)

					crop = localiseplate.ExtractPatch(hsvImg, patchBbox)
					edgeCrop = localiseplate.ExtractPatchGrey(edgeImCombined, patchBbox)

					#feat1 = ExtractHistogram(crop, [np.linspace(0., 1., 10), np.linspace(0., 1., 10), np.linspace(0., 1., 10)])
					edgeFeat = ExtractHistogramGrey(edgeCrop, np.linspace(0., 1500., 100))
					#feats = np.concatenate((feat1, feat2))
					feats = edgeFeat

					#Protect against divide by zero
					feats = np.nan_to_num(feats)

					overlap = OverlapProportion(patchBbox, plateBbox)
					#print cx, cy, overlap, freq
					#if overlap > 0.9:
					#	print cx, cy, overlap, edgeCrop.min(), edgeCrop.max()
					#	misc.imshow(edgeCrop)
					#	plt.plot(feats)
					#	plt.show()

					if overlap == 0.:
						samplesZero.append(feats)
						labelsZero.append(overlap)
						plateIdZero.append(plateId)
					else:
						samplesNonZero.append(feats)
						labelsNonZero.append(overlap)
						plateIdNonZero.append(plateId)

		#Limit number of zero labelled samples
		filtIndex = random.sample(range(len(labelsZero)), maxZeroSamples)
		labelsZeroFilt = [labelsZero[i] for i in filtIndex]
		plateIdZeroFilt = [plateIdZero[i] for i in filtIndex]
		samplesZeroFilt = [samplesZero[i] for i in filtIndex]

		print "Num zero samples", len(samplesZeroFilt), "reduced from", len(samplesZero)
		print "Num non-zero samples", len(samplesNonZero)

		labels.extend(labelsZeroFilt)
		plateIds.extend(plateIdZeroFilt)
		samples.extend(samplesZeroFilt)

		labels.extend(labelsNonZero)
		plateIds.extend(plateIdNonZero)
		samples.extend(samplesNonZero)

	return samples, labels, plateIds

def OverlapProportion(patchBbox, plateBbox):
	ox = OverlapProportion1D(patchBbox[0], patchBbox[1], plateBbox[0], plateBbox[1])
	oy = OverlapProportion1D(patchBbox[2], patchBbox[3], plateBbox[2], plateBbox[3])
	#print patchBbox[2], patchBbox[3], plateBbox[2], plateBbox[3]
	#if oy != 0.:
	#	print ox, oy
	return ox * oy

if __name__ == "__main__":

	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0

	imgPath = None
	if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
		imgPath = sys.argv[1]

	print "Extract features"
	if 1:
		samples, labels, plateIds = GenerateSamples(plates, imgPath)
		samples = np.array(samples)
		pickle.dump((samples, labels, plateIds), open("features.dat", "wb"), protocol=-1)
	
	if 0:
		samples, labels, plateIds = pickle.load(open("features.dat", "rb"))
		print len(labels)

	print "Whiten Features"
	if 1:
		samples = np.array(samples)
		print samples.shape

		var = samples.var(axis=0)
		scaling = np.power(var, 0.5)
		scalingZeros = (scaling == 0.)
		scaling += scalingZeros #Prevent divide by zero
		whitened = samples / scaling
		whitenedVar = whitened.var(axis=0)

		pickle.dump((whitened, labels, plateIds, scaling), open("features-whitened.dat", "wb"), protocol=-1)
	else:
		whitened, labels, plateIds, scaling = pickle.load(open("features-whitened.dat", "rb"))

	print "Extract training data"
	plateIdsSet = set()
	for pi in plateIds:
		plateIdsSet.add(pi)
	plateIdsSet = list(plateIdsSet)

	trainIds = random.sample(plateIdsSet, len(plateIdsSet) / 2)
	trainRows = [(pi in trainIds) for pi in plateIds]
	testRows = [(pi not in trainIds) for pi in plateIds]

	print len(trainRows), sum(trainRows)
	print len(testRows), sum(testRows)
	trainRows = np.array(trainRows, dtype=np.bool)
	testRows = np.array(testRows, dtype=np.bool)

	print "Extract training data"
	trainingData = whitened[trainRows, :]
	trainingLabels = np.array(labels)[trainRows]
	print trainingData.shape

	print "Train regressor"
	#regressor = svm.SVR()
	regressor = ensemble.RandomForestRegressor(n_jobs=4)

	regressor.fit(trainingData, trainingLabels)

	pickle.dump(regressor, open("localise-model.dat", "wb"))

	print "Extract test data"
	
	testData = whitened[testRows, :]
	testLabels = np.array(labels)[testRows]
	predLabels = regressor.predict(testData)
	print testData.shape

	errors = []
	for p, tr in zip(predLabels, testLabels):
		errors.append(p - tr)
	print "Average error", np.abs(errors).mean()
	print "Correlation", np.corrcoef(testLabels, predLabels)[0,1]

	nonZeroLabelErrors = []
	for p, tr in zip(predLabels, testLabels):
		if tr == 0.: continue
		nonZeroLabelErrors.append(p - tr)
	print "Non-zero label error", np.abs(nonZeroLabelErrors).mean()


	plt.plot(testLabels, predLabels, '.')
	plt.show()


