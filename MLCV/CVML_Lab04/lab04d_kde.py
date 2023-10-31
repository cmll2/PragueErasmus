
import argparse
import numpy as np
from numpy.lib.function_base import quantile
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--seed", default=42, type=int, help="Seed for RNG.")
parser.add_argument("--points", default=10, type=int, help="Number of points in the generated data.")
parser.add_argument("--bandwidth", default=0.5, type=float, help="Size of the kernel window.")
parser.add_argument("--kernel", default="tophat", type=str, help="Kernel type for KDE.")
parser.add_argument("--bins", default=25, type=int, help="Number of bins in a histogram.")

def generateData01(generator : np.random.RandomState, N1 : int, N2 : int) -> np.ndarray:
    # Data 1
    data = np.hstack([
        generator.normal(0, 1, N1),
        4 + 3 * generator.normal(0, 1, N2)
    ])

    return np.reshape(data, [-1, 1])

def generateData02(generator : np.random.RandomState, N1 : int, N2 : int, N3 : int) -> np.ndarray:
    # Data 2
    data = np.hstack([
        generator.normal(0, 1, N1),
        4 + generator.normal(0, 1, N2),
        -4 + generator.normal(0, 1, N3)
    ])

    return np.reshape(data, [-1, 1])

def generateData03(generator : np.random.RandomState, N1 : int, N2 : int) -> np.ndarray:
    # Data 3
    data = np.hstack([
        2 * generator.rand(N1),
        generator.triangular(2, 5.5, 7, N2)
    ])
    
    return np.reshape(data, [-1, 1])

def main(args : argparse.Namespace) -> None:
    """
    Parzen window density estimation.
    """
    generator = np.random.RandomState(args.seed)

    # Use the following dataset for probability density estimation.
    # TODO: Try to increase the number of points (the order of magnitude) and compare the results.

    data = [
        generateData01(generator, args.points, args.points),
        generateData02(generator, args.points, args.points, args.points),
        generateData03(generator, args.points, args.points)
    ]

    # Compute histograms
    bins = [ args.bins, args.bins, args.bins ]
    histograms = [
        np.histogram(data[i], bins[i]) for i in range(len(data))
    ]

    # Compute KDE
    # 'kernel' - Type of kernel smoother:
    # 'gaussian' (default), 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'
    ker = args.kernel

    kde = [
        KernelDensity(kernel=ker).fit(data[i]) for i in range(len(data))
    ]
    
    # TODO: Try different values of bandwidth (size of the window - 'h' in the lecture slides, default is 1.0).
    bandwidth = args.bandwidth
    kde_bw = [
        KernelDensity(kernel=ker, bandwidth=bandwidth).fit(data[i]) for i in range(len(data))
    ]

    # TODO: Compute suggested values of bandwidth.
    # IQR
    y = [
        quantile(data[i], [0.25, 0.75]) for i in range(len(data))
    ]
    y = np.asarray(y)
    IQR = y[:, 1] - y[:, 0]

    # TODO: Compute the standard deviation for all datasets and name it 'sig'.
    # - The result should be a list similar to 'y'.
    sig = [np.std(data[i]) for i in range(len(data))]

    # TODO: Compute the multi-modal bandwidth for all datasets and name it 'multi_bw'.
    # - Bandwidth is called 'h' in the lecture slides.
    # - The result should be a list similar to 'y'.
    multi_bw = [0.9 * min(sig[i], IQR[i] / 1.34) * len(data[i]) ** (-1/5) for i in range(len(data))]

    # TODO: Compute the unimodal bandwidth for all datasets and name it 'uni_bw'.
    # - Bandwidth is called 'h' in the lecture slides.
    # - The result should be a list similar to 'y'.
    uni_bw = [1.06 * sig[i] * len(data[i]) ** (-1/5) for i in range(len(data))]


    if multi_bw is not None:
        kde_iqr = [
            KernelDensity(kernel=ker, bandwidth=multi_bw[i]).fit(data[i]) for i in range(len(data))
        ]
    if uni_bw is not None:
        kde_gauss = [
            KernelDensity(kernel=ker, bandwidth=uni_bw[i]).fit(data[i]) for i in range(len(data))
        ]

    # Display the data and density estimates.
    fig, ax = plt.subplots(1, len(data), figsize=(6 * len(data), 6), subplot_kw={'aspect': 'auto'})
    for i in range(len(data)):
        samples = np.linspace(np.min(data[i]) - 1, np.max(data[i]) + 1, 50)
        ax[i].bar(histograms[i][1][:-1], histograms[i][0], histograms[i][1][1] - histograms[i][1][0], align="edge", label="Data")
        
        scores = np.exp(kde[i].score_samples(np.reshape(samples, [-1, 1])))
        scores = np.max(histograms[i][0]) * (scores - np.min(scores)) / np.max(scores)
        ax[i].plot(samples, scores, color="red", label="Base KDE")

        scores_bw = np.exp(kde_bw[i].score_samples(np.reshape(samples, [-1, 1])))
        scores_bw = np.max(histograms[i][0]) * (scores_bw - np.min(scores_bw)) / np.max(scores_bw)
        ax[i].plot(samples, scores_bw, color="blue", label="KDE + custom bw")

        if multi_bw is not None:
            scores_iqr = np.exp(kde_iqr[i].score_samples(np.reshape(samples, [-1, 1])))
            scores_iqr = np.max(histograms[i][0]) * (scores_iqr - np.min(scores_iqr)) / np.max(scores_iqr)
            ax[i].plot(samples, scores_iqr, color="green", label="KDE + multi-modal bw")

        if uni_bw is not None:
            scores_gauss = np.exp(kde_gauss[i].score_samples(np.reshape(samples, [-1, 1])))
            scores_gauss = np.max(histograms[i][0]) * (scores_gauss - np.min(scores_gauss)) / np.max(scores_gauss)
            ax[i].plot(samples, scores_gauss, color="black", label="KDE + unimodal bw")
        
        ax[i].legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
