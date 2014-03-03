pyanpr-imagemine
================

Tests of automatic number plate recognition in python.

==Required Python Libraries==

Required:
Python 2.7 or similar.
Numpy (http://www.numpy.org/)
Scipy (http://scipy.org/)
skimage (http://scikit-image.org/)
Python Imaging Library (http://www.pythonware.com/products/pil/)

==Training==

NOTE: The images may need to be in a specific folder. More work is needed on the code to enable this to be specified on the command line. The git repo contains a complete model, so training is optional.

The original number plates were annotated using imgAnnotation (https://github.com/alexklaeser/imgAnnotation) and saved in the plates.annotation file.

Run deskewMarkedPlates.py to horizontally align the number plate images. This saves alignment data into the "train" sub folder.

managetrainingchars.py is an interactive, command line tool to split the plates into separate characters and verify the character bounding boxes match the true registration. Saving data from this tool generates the files:

charbboxes.dat
charcofgs.dat
charbboxangle.dat
charstrings.dat

Run trainchars.py to collect examples of each character. Only half the data is used in training and the remainer is reserved for testing. This produces the file: charmodel.dat

==ANPR Recognition==

The main script to recognise plates is anpr.py. Run this without arguments and it begins processing the unseen test data. It is likely the image files are stored in a different location than the program expects, so pass the folder location to anpr.py and it will process unseen test images.

python anpr.py anpr-plates/

Alternatively, specify a single image to process as the first argument:

python anpr.py /path/to/file/IMG_20140219_105154.jpeg


