from matplotlib import pyplot as plt
import numpy as np

def plotData(data, typ):
    """plot y=intensity, x=2theta"""
    filename = f"../figures/{typ}.pdf"
    fig = plt.figure()
    plt.plot(data[:, 0], data[:, 1], color = "black")
    plt.xlabel(r"2$\theta$")
    plt.ylabel(r"$Intensity$")
    plt.savefig(filename, dpi=600)
    plt.show()

def plotIntegral(data, a, b, typ, trapezoidals = False):
    """ plot that shows the integrated area,
        if trapezoidals = True -> trapezoidal
        area will show"""
    filename = f"../figures/{typ}_integral_{round(a)},{round(b)}.pdf"

    fig = plt.figure()
    plt.xlabel(r"2$\theta$")
    plt.ylabel(r"$Intensity$")

    x = np.append(data[a:b+1, 0], np.flip(data[a:b+1, 0]))
    y = np.append(data[a:b+1, 1], np.zeros(b-a+1))

    plt.plot(data[:, 0], data[:, 1], color = "black", linewidth = 1)
    plt.fill(x, y, color = "grey")

    if trapezoidals:
        i = a
        while i < b:
            plt.plot([data[i, 0], data[i, 0], data[i+1, 0], data[i+1, 0]],
                     [0.0, data[i, 1], data[i+1, 1], 0.0], color = "green")
            plt.plot([data[i, 0], data[i, 0], data[i+1, 0], data[i+1, 0]],
                     [0.0, data[i, 1], data[i+1, 1], 0.0], "o", color = "green")
            i += 1

    plt.savefig(filename, dpi=600)
    plt.show()

def compositeTrapezoidalIntegration(data, a, b):
    """ returns trapezoidal integral between a, b
    finds nearest x values to a and b where b > a"""
    aIdx = (np.abs(data[:, 0] - a)).argmin()
    bIdx = (np.abs(data[:, 0] - b)).argmin()
    T = 0.
    for i in range(aIdx, bIdx):
        T += 0.5 * (data[i+1, 0] - data[i, 0]) * (data[i+1, 1] + data[i, 1])
    return T, aIdx, bIdx

def simpsonsIntegration(data, a, b):
    """ returns simpsons integral between a, b
    finds nearest x values to a and b where b > a"""
    aIdx = (np.abs(data[:, 0] - a)).argmin()
    bIdx = (np.abs(data[:, 0] - b)).argmin()
    cutData = data[aIdx:bIdx+1, :]
    if (bIdx - aIdx + 1) % 2 == 0: bIdx += 1 # assert odd number of points

    T = data[aIdx, 1] + data[bIdx, 1]
    for i in range(aIdx + 1, bIdx):
        if i % 2 == 0:
            T += 2. * data[i, 1]
        else:
            T += 4. * data[i, 1]
    T *= (data[bIdx, 0] - data[bIdx - 1, 0]) / 3.
    return T, aIdx, bIdx

def compositeSimpsonsIntegration(data, a, b):
    """ returns composite simpsons integral between a, b
    finds nearest x values to a and b where b > a"""
    aIdx = (np.abs(data[:, 0] - a)).argmin()
    bIdx = (np.abs(data[:, 0] - b)).argmin()

    x, y = data[aIdx:bIdx+1, 0], data[aIdx:bIdx+1, 1]
    points = len(x) - 1
    dx = np.diff(x)

    T = 0.
    for i in range(1, points, 2):
        h2 = dx[i] + dx[i-1]
        T += y[i] * (dx[i]**3 + dx[i-1]**3 + 3. * dx[i] * dx[i-1] * h2) / (6. * dx[i] * dx[i-1]) + y[i-1] * (2. * dx[i-1]**3 - dx[i]**3 + 3. * dx[i] * dx[i-1]**2) / (6. * dx[i-1] * h2) + y[i+1] * (2. * dx[i]**3 - dx[i-1]**3 + 3. * dx[i-1] * dx[i]**2) / (6. * dx[i] * h2)

    if (points + 1) % 2 == 0:
        T += y[points] * (2. * dx[points-1]**2 + 3. * dx[points-2] * dx[points-1]) / (6. * (dx[points-2] + dx[points-1])) + y[points-1] * (dx[points-1]**2 + 3. * dx[points-1]*dx[points-2]) / (6. * dx[points-2]) - y[points-2] * dx[points-1]**3 / (6. * dx[points-2] * (dx[points-2] + dx[points-1]))

    return T, aIdx, bIdx

def performIntegration(data, idx, f, typ):
    """integrate data from idx[:, 0]->idx[:, 1]
       with f method. print result"""
    integrals = np.zeros(len(idx))
    for i in range(len(idx)):
        a, b = idx[i, :]
        integrals[i], a, b = f(data, a, b)
        plotIntegral(data, a, b, typ)
    
    print("type : {typ}")
    for i in range(len(integrals)):
        print(f"I{i} = {integrals[i]}")
    print("---------")
    print(f"I2/I1 = {integrals[1]/integrals[0]*100} %")
    print(f"I3/I1 = {integrals[2]/integrals[0]*100} %")
    print("========\n")

if __name__ == "__main__":
    # read data
    dataSi = np.loadtxt("../data/scan1.txt")
    dataUnknown = np.loadtxt("../data/scan2.txt")
    dataGauss = np.loadtxt("../data/curveFitGaussians.txt")

    plotData(dataSi, "si")
    plotData(dataUnknown, "unknown")
    plotData(dataGauss, "gauss")
    
    # indexes used by dataUnknown integration
    kclIdx = np.array([
        [12.5, 13.5],
        [17.8, 18.8],
        [21.9, 22.9]])
    naclIdx = np.array([
        [14., 14.75],
        [20.1, 21.],
        [24.7, 25.6]])
    performIntegration(dataUnknown, naclIdx, compositeSimpsonsIntegration, "unknown")
    
    # indexes used by dataGAuss integration
    kclIdx = np.array([
        [12.2, 13.8],
        [17.5, 19.0],
        [21.4, 23.9]])
    naclIdx = np.array([
        [13.5, 14.95],
        [19.5, 21.5],
        [24.7, 25.6]])
    performIntegration(dataGauss, naclIdx, compositeSimpsonsIntegration, "gauss")

    # indexes used by si integration
    #siIdx = np.array([
    #    [11.93, 13.86],
    #    [20.47, 21.9],
    #    [24.38, 25.59]])
    #performIntegration(dataSi, siIdx, trapezoidalIntegration, "si")

    #siIdx = np.array([
    #    [12.25, 13.25],
    #    [20.75, 21.75],
    #    [24.37, 25.5]])
    #performIntegration(dataSi, siIdx, trapezoidalIntegration, "si")
