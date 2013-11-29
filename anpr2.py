import pickle, sys
import scipy.misc as misc

if __name__=="__main__":

	fina = None
	if len(sys.argv) >= 2:
		fina = sys.argv[1]
	if fina is None:
		print "Specify input image on command line"
		exit(0)
	im = misc.imread(fina)

	roi = pickle.load(open("out.dat","rb"))
	bbox = roi[2]
	print bbox

	im = im[bbox[1][0]:bbox[1][1],:]
	im = im[:,bbox[0][0]:bbox[0][1]]

	im = misc.imsave("test.png", im)

