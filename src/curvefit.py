from matplotlib import pyplot as plt
from lmfit import models, Parameters, Model
from lmfit.model import save_modelresult
import numpy as np

class CurveFit:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.count = 0
        self.params = Parameters()
        self.result = None

    def addGaussian(self, center, sigma, height, minn, maxx):
        gaussModel = models.GaussianModel(prefix = f"m{self.count}")
        gaussModel.set_param_hint("center", min=minn, max=maxx)
        gaussModel.set_param_hint("sigma", min=1e-4, max=np.max(data[:, 0]))
        gaussParams = gaussModel.make_params(center=center, sigma=sigma, 
                                             height=height)

        if self.model != None: self.model += gaussModel
        else: self.model = gaussModel
        self.params.update(gaussParams)
        self.count += 1

    def addVoigt(self, center, sigma, height, gamma, minn, maxx):
        voigtModel = models.VoigtModel(prefix = f"m{self.count}")
        voigtModel.set_param_hint("center", min=minn, max=maxx)
        voigtModel.set_param_hint("sigma", min=1e-4, max=np.max(data[:, 0]))
        voigtParams = voigtModel.make_params(center=center, sigma=sigma, 
                                             height=height, gamma=gamma)

        if self.model != None: self.model += voigtModel
        else: self.model = voigtModel
        self.params.update(voigtParams)
        self.count += 1

    def execute(self, weights):
        self.result = self.model.fit(self.data[:, 1], self.params, 
                                     x=self.data[:, 0], weights=weights)

    def plotResult(self):
        assert(self.result != None)
        fig = plt.figure()
        plt.scatter(self.data[:, 0], self.data[:, 1], color = "grey", label = "data points")
        plt.plot(self.data[:, 0], self.result.best_fit, color = "blue", label = "curve fit")
        plt.legend()
        plt.show()

        fig, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1])
        models = self.result.eval_components(x=self.data[:, 0])
        for i in range(self.count):
            ax.plot(self.data[:, 0], models[f"m{i}"], label=f"m{i}")
        plt.legend()
        plt.show()

    def saveResult(self, filename="../data/curveFitResult.txt"):
        assert(self.result != None)
        save_modelresult(self.result, filename)

if __name__ == "__main__":
    data = np.loadtxt("../data/scan2.txt")

    cutXLow, cutXHigh = 10., 28.
    cutIdxLow = (np.abs(data[:, 0] - cutXLow)).argmin()
    cutIdxHigh = (np.abs(data[:, 0] - cutXHigh)).argmin()
    data = data[cutIdxLow:cutIdxHigh, :]

    curveFit = CurveFit(data)

    # initialize noise functions
    curveFit.addVoigt(10.3, 5.1, 50., 0.5, 10., 20.)
    curveFit.addVoigt(22., 0.1, 25., 0.1, 10., 25.)
    curveFit.addVoigt(25., 1.1, 20., 4.1, 20., 25.)
    curveFit.addVoigt(17., 2.1, 10., 4.1, 10., 25.)
    curveFit.addVoigt(18., 0.1, 20., 2.1, 10., 25.)

    # initialize top functions
    curveFit.addGaussian(12.90, 0.1, 225., 12.5, 13.3)
    curveFit.addGaussian(14.38, 0.1, 329.8, 14.11, 14.81)
    curveFit.addGaussian(18.36, 0.1, 147., 18., 18.6)
    curveFit.addGaussian(20.5, 0.1, 188., 20.1, 20.8)
    curveFit.addGaussian(22.50, 0.1, 25.8, 22.2, 22.8)
    curveFit.addGaussian(25.1, 0.2, 75.6, 24.65, 25.5)

    # manual adjust height of top peak
    weights = np.ones(len(curveFit.data[:, 1]))
    weights[curveFit.data[:, 1].argmax()] *= 4
    
    # fit, plot and save data
    curveFit.execute(weights)
    curveFit.plotResult()
    curveFit.saveResult()
