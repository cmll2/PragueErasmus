
import load_mnist
from matplotlib import pyplot as plt
import numpy as np

def main():

    # 1. Use 'load_mnist.loadMnist' function to read 60000 training images and class labels
    #    of MNIST digits (files 'train-images.idx3-ubyte' and 'train-labels.idx1-ubyte').
    
    images, labels = load_mnist.loadMnist('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')

    # 2. Preview one image from each class.
    
    unique_labels = np.unique(labels)

    for label in unique_labels:

        index = np.where(labels == label)[0][0]

        image = images[index].reshape(28, 28)  # Les images MNIST sont de taille 28x28 pixels

        plt.subplot(2, 5, label + 1)  # 2 lignes, 5 colonnes pour afficher 10 classes
        plt.imshow(image, cmap='gray')
        plt.title(f'Class {label}')

    # Affichez les images
    plt.tight_layout()
    plt.show()

    # 3. Transform the image data, such that each image forms a row vector,
    #    - NOTE: Math in lectures assumes the column-format, exercises will assume the row-format
    #      (Row-format is used by most Python libraries). This means that we will have to "transpose"
    #      formulae before using them.

    num_images = images.shape[0]
    image_size = images.shape[1] * images.shape[2]

    flattened_images = images.reshape(num_images, image_size)
    
    # 4. Save the image matrix and the labels in a numpy .npy or .npz files.
    #    'np.save'/'np.savez'/'np.savez_compressed', loading is done using 'np.load'
    #    - 'np.savez_compressed' is the most efficient.

    np.savez_compressed('mnist.npz', images=flattened_images, labels=labels)
    
    # 5. Do the same for 10000 test digits (files 't10k-images.idx3-ubyte' and
    #    't10k-labels.idx1-ubyte')
    #    - Both files (training and testing set) will be used during the semester.

    test_images, test_labels = load_mnist.loadMnist('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

    flattened_test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

    np.savez_compressed('t10k.npz', images=flattened_test_images, labels=test_labels)
    
    # 6. Now, try to load the created files, display some of the images and print their respective
    #    labels.
    
    data = np.load('mnist.npz')
    flattened_images = data['images']
    labels = data['labels']

    # Charger les fichiers de test
    test_data = np.load('t10k.npz')
    flattened_test_images = test_data['images']
    test_labels = test_data['labels']

    num_images_to_display = 5  

    # Créez une figure pour afficher les images
    plt.figure(figsize=(12, 5))

    for i in range(num_images_to_display):
        # Affichez l'image
        plt.subplot(1, num_images_to_display, i + 1)
        plt.imshow(flattened_images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {labels[i]}')

    # Affichez les images
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
