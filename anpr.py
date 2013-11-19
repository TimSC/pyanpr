
import scipy.misc as misc
import scipy.signal as signal
import numpy as np

if __name__ == "__main__":
	im = misc.imread("../pyanpr-data/56897161_d613d63bce_b.jpg")
	greyim = 0.2126 * im[:,:,0] + 0.7152 * im[:,:,1] + 0.0722 * im[:,:,2]

	vkernel = np.array([[-0.5,-1.,-0.5],[0.,0.,0.],[0.5,1.,0.5]])
	filterim = signal.convolve2d(greyim, vkernel, mode="same")
	misc.imsave("vim.png", filterim)

	hkernel = np.array([[-0.5,0.,0.5],[-1,0.,1],[-0.5,0,0.5]])
	filterim2 = signal.convolve2d(greyim, hkernel, mode="same")
	misc.imsave("him.png", filterim2)

