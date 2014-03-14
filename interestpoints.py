
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

#50 bin 2D absolute sobel
#Average error 0.0464
#Correlation 0.51
#Non-zero label error 0.478

#Fused HSV and 50 bin 2d absolute sobel
#Average error 0.0405
#Correlation 0.59
#Non-zero label error 0.437


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

					hsvFeat = ExtractHistogram(crop, [np.linspace(0., 1., 10), np.linspace(0., 1., 10), np.linspace(0., 1., 10)])
					edgeFeat = ExtractHistogramGrey(edgeCrop, np.linspace(0., 1500., 50))
					#feats = np.concatenate((feat1, feat2))
					feats = [hsvFeat, edgeFeat]
					#print feats

					#Protect against divide by zero
					feats = map(np.nan_to_num, feats)

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

def SplitData(trainIds, plateIds, labels, whitenedSamples):
	trainRows = [(pi in trainIds) for pi in plateIds]
	testRows = [(pi not in trainIds) for pi in plateIds]

	print len(trainRows), sum(trainRows)
	print len(testRows), sum(testRows)
	trainRows = np.array(trainRows, dtype=np.bool)
	testRows = np.array(testRows, dtype=np.bool)

	trainingData = [table[trainRows, :] for table in whitenedSamples]
	trainingLabels = np.array(labels)[trainRows]
	trainingPlateIds = np.array(plateIds)[trainRows]

	testData = [table[testRows, :] for table in whitenedSamples]
	testLabels = np.array(labels)[testRows]
	testPlateIds = np.array(plateIds)[testRows]

	return trainingData, trainingLabels, trainingPlateIds, \
		testData, testLabels, testPlateIds

if __name__ == "__main__":

	plates = readannotation.ReadPlateAnnotation("plates.annotation")
	count = 0

	imgPath = None
	if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
		imgPath = sys.argv[1]

	print "Extract features"
	if 0:
		samples, labels, plateIds = GenerateSamples(plates, imgPath)
		samplesArray = []
		for featGroupNum in range(len(samples[0])):
			featGroup = [s[featGroupNum] for s in samples]
			sampleGroup = np.array(featGroup)
			samplesArray.append(sampleGroup)
		pickle.dump((samplesArray, labels, plateIds), open("features.dat", "wb"), protocol=-1)

	if 1:
		samples, labels, plateIds = pickle.load(open("features.dat", "rb"))
		print len(labels), len(samples)

	print "Whiten Features"
	if 1:
		whitenedSamples = []
		for featGroup in samples:

			featGroup = np.array(featGroup)
			print featGroup.shape

			var = featGroup.var(axis=0)
			scaling = np.power(var, 0.5)
			scalingZeros = (scaling == 0.)
			scaling += scalingZeros #Prevent divide by zero
			whitened = featGroup / scaling
			#whitenedVar = whitened.var(axis=0)
			whitenedSamples.append(whitened)

		pickle.dump((whitenedSamples, labels, plateIds, scaling), open("features-whitened.dat", "wb"), protocol=-1)
	else:
		whitenedSamples, labels, plateIds, scaling = pickle.load(open("features-whitened.dat", "rb"))

	print "Plan train and test split data"
	plateIdsSet = set()
	for pi in plateIds:
		plateIdsSet.add(pi)
	plateIdsSet = list(plateIdsSet)
	trainIds = random.sample(plateIdsSet, int(round(len(plateIdsSet) * 0.5)))

	trainingData, trainingLabels, trainingPlateIds, \
		testData, testLabels, testPlateIds = SplitData(trainIds, plateIds, labels, whitenedSamples)

	trainIds1 = random.sample(trainIds, int(round(len(trainIds) * 0.8)))
	trainingData1, trainingLabels1, trainingPlateIds1, \
		trainingData2, trainingLabels2, trainingPlateIds2 = SplitData(trainIds1, trainingPlateIds, trainingLabels, trainingData)

	print trainingLabels1.shape
	print trainingLabels2.shape

	print "Train regressor"
	#regressor = svm.SVR()
	models = []
	for featureGroup in trainingData1:
		regressor = ensemble.RandomForestRegressor(n_jobs=4)
		regressor.fit(featureGroup, trainingLabels1)
		models.append(regressor)

	print "Train fusion model"
	predLabels = []
	for featureGroup, regressor in zip(trainingData2, models):
		predLabelGroup = regressor.predict(featureGroup)
		predLabels.append(np.array(predLabelGroup).reshape(len(predLabelGroup), 1))
	fusedFeatures = None
	for predLabelGroup in predLabels:
		if fusedFeatures is None:
			fusedFeatures = predLabelGroup
		else:
			fusedFeatures = np.hstack((fusedFeatures, predLabelGroup))
	for featureGroup in trainingData2:
		fusedFeatures = np.hstack((fusedFeatures, featureGroup))
	fusionModel = ensemble.RandomForestRegressor(n_jobs=4)
	fusionModel.fit(fusedFeatures, trainingLabels2)

	pickle.dump((models, fusionModel), open("localise-model.dat", "wb"))

	print "Predict intermediate labels on test data"
	predLabels = []
	for featureGroup, regressor in zip(testData, models):
		predLabelGroup = regressor.predict(featureGroup)
		predLabels.append(np.array(predLabelGroup).reshape(len(predLabelGroup), 1))

	print "Fuse intermediate results"

	fusedFeatures = None
	for predLabelGroup in predLabels:
		if fusedFeatures is None:
			fusedFeatures = predLabelGroup
		else:
			fusedFeatures = np.hstack((fusedFeatures, predLabelGroup))
	for featureGroup in testData:
		fusedFeatures = np.hstack((fusedFeatures, featureGroup))

	predLabels = fusionModel.predict(fusedFeatures)

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


