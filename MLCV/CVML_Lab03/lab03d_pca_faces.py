
import os
import re
import argparse
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import lab03b_pca_svd

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--path", default="./centered/", type=str, help="Path to the 'face' dataset with root directory 'centered'.")
parser.add_argument("--compare_idx", default=22, type=int, help="Index of the iamge selected for comparison.")
parser.add_argument("--reconstruct_idx", default=11, type=int, help="Index of the image selected for gradual reconstruction.")

def main(args : argparse.Namespace) -> None:
    # TODO: Make sure that 'args.path' contains a correct path towards folder 'centered' containing
    # face dataset downloaded from moodle.
    training_folder = args.path
    regex = re.compile('.*\..*\.pgm$')
    train_files = []
    dirs = os.listdir(args.path)
    for f in dirs:
        if regex.match(f):
            train_files.append(f)
    train_files = [training_folder + f for f in train_files]

    # Get the resolution of the images (from the first one).
    im = skimage.io.imread(train_files[0])
    H = im.shape[0]
    W = im.shape[1]
    # How many images are there?
    M = len(train_files)

    # Read all images and transform them to vectors (create a matrix and name it X).
    flattened_imgs = np.zeros([M, H * W])
    for i in range(M):
        flattened_imgs[i] = skimage.io.imread(train_files[i]).flatten()

    # TODO: Compute eigenvectors forming the matrix B using SVD.
    # - you can use either 'svdPCA' which you implemented in 'lab03b_pca_svd.py'
    #   or compute it again using 'np.linalg.svd'

    # Name the variables as follows (so that the rest of the code works):
    # mx - the mean of the data
    # B - the eigenvectors in row form (1 row = 1 eigenvector)
    # X - the centred data (centred 'flattened_imgs')
    # S - explained variances

    B, S, mx, projected = lab03b_pca_svd.svdPCA(flattened_imgs)
    B = B.T
    X = flattened_imgs - mx
    # TODO: Show one of the eigenfaces.
    pc = 0
    fig, ax = plt.subplots(1, 1, figsize=(6, 7))
    ax.set_title("PC {} reshaped back into an image".format(pc))
    ax.imshow(np.reshape(B[pc, :], [H, W]), cmap='gray')
    fig.tight_layout()
    plt.show()

    # TODO: Determine the number of PCs to keep, look at the cumulative variances.
    # - Either plot the graph and read the value from there or use an automatic method.
    threshold = 0.95
    cumsum = np.cumsum(S/np.sum(S))
    k = np.argmax(cumsum > threshold) + 1
    
    # Keep k principal components.
    Bk = B[:k, :]

    # Compute the new coordinates in k dimensions.
    Xk = X @ Bk.T

    # Reconstruct the reduced data back to D dimensions.
    rec = mx + Xk @ Bk

    # Compare the original image and the reconstruction from principal components.
    idx = args.compare_idx
    fig, ax = plt.subplots(1, 2, subplot_kw={'aspect': 'equal'})
    ax[0].set_title("Original image")
    ax[0].imshow(np.reshape(flattened_imgs[idx, :], [H, W]), cmap='gray')
    ax[1].set_title("Reconstruction from {} PCs".format(k))
    ax[1].imshow(np.reshape(rec[idx, :], [H, W]), cmap='gray')
    fig.tight_layout()
    plt.show()

    # Now, let's visualise the reconstructions from different number of principal components.
    # You can change the value of 'imgIdx' to view the reconstruction of other images.
    # Also, you can modify the 'counts' list to see the reconstructions which interest
    # you.
    imgIdx = args.reconstruct_idx
    counts = [1, 2, 5, 10, 20, 40, 80, 120]
    fig, ax = plt.subplots(1, len(counts) + 1, figsize=(20, 3), subplot_kw={'aspect': 'equal'})
    for i in range(len(counts)):
        count = counts[i]
        projected = X[imgIdx, :] @ B[:count, :].T
        reconstructed = mx + projected @ B[:count, :]
        ax[i].imshow(np.reshape(reconstructed, [H, W]), cmap='gray')
        ax[i].set_axis_off()
        ax[i].set_title("{} PCs".format(count))
    ax[len(counts)].imshow(np.reshape(flattened_imgs[imgIdx, :], [H, W]), cmap='gray')
    ax[len(counts)].set_axis_off()
    ax[len(counts)].set_title("Original")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
