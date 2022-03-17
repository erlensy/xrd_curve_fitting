from matplotlib import pyplot as plt
from scipy import special
from lmfit.model import load_modelresult
import numpy as np

def voigt(x, amplitude, center, sigma, gamma):
    z = (x - center + 1j*gamma) / (sigma*np.sqrt(2))
    w = np.exp(-z*z) * special.erfc(-1j*z)
    return amplitude * np.real(w) / (sigma * np.sqrt(2*np.pi))

def gaussian(x, amplitude, center, sigma):
    k = np.exp(-((x-center)**2)/(2*sigma*sigma))
    return amplitude / (sigma*np.sqrt(2*np.pi)) * k

def readCurveFitResult(filename = "../data/curveFitResult.txt"):
    curveFit = load_modelresult(filename)
    gaussians = np.zeros((6, 3)) # amplitude, center, sigma
    voigts = np.zeros((5, 4)) # amplitude, center, sigma, gamma
    gaussianPrefixes = np.array(["m5", "m6", "m7", "m8", "m9", "m10"])
    i, j, k, l = 0, 0, 0, 0
    for key in curveFit.best_values:
        if key[:3] in gaussianPrefixes or key[:2] in gaussianPrefixes:
            gaussians[i, j] = curveFit.best_values[key]
            if j == 2: i += 1; j = 0
            else: j += 1
        else:
            voigts[k, l] = curveFit.best_values[key]
            if l == 3: k += 1; l =0
            else: l += 1
    return gaussians, voigts

def plotCurveFit(data):
    gaussians, voigts = readCurveFitResult()
    n = 1000
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), n)
    yGauss = np.zeros((n, len(gaussians)))
    yVoigt = np.zeros((n, len(voigts)))
    for i in range(n):
        for j in range(len(gaussians)):
            yGauss[i, j] = gaussian(x[i], gaussians[j, 0], gaussians[j, 1], gaussians[j, 2])

        for j in range(len(voigts)):
            yVoigt[i, j] = voigt(x[i], voigts[j, 0], voigts[j, 1], voigts[j, 2], voigts[j, 3])
    
    yTot = np.zeros(n)
    for i in range(len(gaussians)):
        yTot += yGauss[:, i]
    for i in range(len(voigts)):
        yTot += yVoigt[:, i]
    fig = plt.figure()
    plt.grid()
    plt.scatter(data[:, 0], data[:, 1], color = "grey")
    plt.plot(x, yTot)
    plt.show()

def plotCurveFitFunctions(data):
    gaussians, voigts = readCurveFitResult()
    n = 1000
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), n)
    yGauss = np.zeros((n, len(gaussians)))
    yVoigt = np.zeros((n, len(voigts)))
    for i in range(n):
        for j in range(len(gaussians)):
            yGauss[i, j] = gaussian(x[i], gaussians[j, 0], gaussians[j, 1], gaussians[j, 2])
        for j in range(len(voigts)):
            yVoigt[i, j] = voigt(x[i], voigts[j, 0], voigts[j, 1], voigts[j, 2], voigts[j, 3])

    fig = plt.figure()
    plt.grid()
    plt.scatter(data[:, 0], data[:, 1], color = "grey", label = "data", s = 12)

    for i in range(len(voigts)):
        label = "voigt profiles" if i == 0 else "__nolegend__"
        plt.plot(x, yVoigt[:, i], color = "blue", label = label)
    for i in range(len(gaussians)):
        label = "gaussian functions" if i == 0 else "__nolegend__"
        plt.plot(x, yGauss[:, i], color = "red", label = label)

    plt.legend()
    plt.show()

def saveCurveFitData(data):
    gaussians, voigts = readCurveFitResult()
    n = 1000
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), n)
    yGauss = np.zeros((n, len(gaussians)))
    for i in range(n):
        for j in range(len(gaussians)):
            yGauss[i, j] = gaussian(x[i], gaussians[j, 0], gaussians[j, 1], gaussians[j, 2])
    gaussTot = np.zeros((n, 2))
    gaussTot[:, 0] = x
    for i in range(len(gaussians)):
        gaussTot[:, 1] += yGauss[:, i]

    np.savetxt("../data/curveFitGaussians.txt", gaussTot)

if __name__ == "__main__":
    data = np.loadtxt("../data/scan2.txt")
    cutXLow, cutXHigh = 10., 28.
    cutIdxLow = (np.abs(data[:, 0] - cutXLow)).argmin()
    cutIdxHigh = (np.abs(data[:, 0] - cutXHigh)).argmin()
    data = data[cutIdxLow:cutIdxHigh, :]

    plotCurveFit(data)
    plotCurveFitFunctions(data)
    saveCurveFitData(data)
