import numpy as np


def gabor2d(gw, gh, gx0, gy0, wfreq, worient, wphase, shape):
    """ Generate a gabor 2d array

    Inputs:
      gw -- width of the gaussian envelope
      gh -- height of the gaussian envelope
      gx0 -- x indice of center of the gaussian envelope
      gy0 -- y indice of center of the gaussian envelope
      wfreq -- frequency of the 2d wave
      worient -- orientation of the 2d wave
      wphase -- phase of the 2d wave
      shape -- shape tuple (height, width)

    Outputs:
      gabor -- 2d gabor with zero-mean and unit-variance

    """

    height, width = shape
    y, x = np.mgrid[0:height, 0:width]

    X = x * np.cos(worient) * wfreq
    Y = y * np.sin(worient) * wfreq

    env = np.exp( -np.pi * ( ((x-gx0)**2./gw**2.) + ((y-gy0)**2./gh**2.) ) )
    wave = np.exp( 1j*(2*np.pi*(X+Y) + wphase) )
    gabor = np.real(env * wave)

    gabor -= gabor.mean()
    gabor /= fastnorm(gabor)

    return gabor


def fastnorm(x):
    """ Fast Euclidean Norm (L2)
    """
    xv = x.ravel()
    return np.dot(xv, xv)**(1/2.)