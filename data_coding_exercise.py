# data_coding_exercise.py
#
# This script will read in a featurized dataset of samples and
# find the first two principle components. It will then plot the
# negative log of the probability along each as well as a combined
# contour plot showing the realtion between both. The amount of
# variance in the original data that is explained by these first
# two principal components is also printed as output. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_and_prepare_data(fname):
    # Load the text file in as a pandas dataframe
    dataframe = pd.read_csv(fname, sep=' ', header=None)
    print("The dataframe has",dataframe.shape[0],"rows and",dataframe.shape[1],"columns.")

    # Subtract means from each column to get zero-centered features
    features = dataframe.copy()
    features_zero_centered = features.sub(features.mean())

    return features_zero_centered.values


def pca_manual(A):

    # generate covariance matrix for featurized, mean-centered data
    cov_matrix = np.cov(A.T)

    # calculate eigenvectors and eigenvalues
    evals, evects = np.linalg.eig(cov_matrix)
    
    # sort data in order of descending eigenvalues
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evects = evects[:,order]
 
    # print how much of data each PC explains
    print("PC1 explains", str(((evals/np.sum(evals))*100)[0])+"%", "of the variance.")
    print("PC2 explains", str(((evals/np.sum(evals))*100)[1])+"%", "of the variance.")
    
    # transform original data
    transformed_data = evects.T.dot(A.T) 

    # extract first and second principal components
    pc1 = transformed_data[0]
    pc2 = transformed_data[1]

    return pc1, pc2


def plot_1d_hist(pc, pc_name, bins=10):
    plt.clf()

    # Plot -ln(P) of pc1
    n, bin_edges = np.histogram(pc, bins)
    bin_probability = -1*np.log(n/float(n.sum()))
    bin_probability -= np.min(bin_probability)
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2
    bin_width = bin_edges[1]-bin_edges[0]
    plt.bar(bin_middles, bin_probability, width=bin_width, facecolor="dodgerblue", alpha=0.7)
    plt.plot(bin_middles, bin_probability, linewidth=1.5, color="k")
    plt.ylim(0,0.3)
    plt.xlabel(pc_name)
    plt.ylabel("-ln(P)")
    plt.title("1D Histogram of "+pc_name)
    plt.savefig(pc_name+"_hist.pdf")

def plot_2d_hist(pc1, pc2, pc1_name, pc2_name, bins=10):
    plt.clf()

    # Plot pc1 vs pc2 as a Contour Plot
    n, bin_xedges, bin_yedges = np.histogram2d(pc1, pc2, bins)
    bin_probability = -1*np.log(n/float(n.sum()))
    bin_probability -= np.min(bin_probability)
    bin_xmiddles = (bin_xedges[1:]+bin_xedges[:-1])/2
    bin_ymiddles = (bin_yedges[1:]+bin_yedges[:-1])/2
    plt.contourf(bin_xmiddles, bin_ymiddles, bin_probability, 20, cmap="jet")
    plt.xlabel(pc1_name)
    plt.ylabel(pc2_name)
    plt.title("2D Histogram of "+pc1_name+" and "+pc2_name)
    plt.colorbar(label="-ln(P)")
    plt.savefig(pc1_name+"_"+pc2_name+"_hist.pdf")

if __name__ == "__main__":
    # Load and prepare data
    data = load_and_prepare_data("data_coding_exercise.txt")
    
    # sanity check against scikitlearn's PCA calculation
    pca = PCA(n_components=2)
    pca.fit(data)
    tdata = pca.transform(data)

    # Get the first and second principal components manually
    pc1, pc2 = pca_manual(data)

    # check that my PCs are close to those from scikitlearn
    assert (tdata[:,0][0] - pc1[0]) < 1E-10
    
    # Make plots
    plot_1d_hist(pc1, "pc1")
    plot_1d_hist(pc2, "pc2")
    plot_2d_hist(pc1, pc2, "pc1", "pc2")
