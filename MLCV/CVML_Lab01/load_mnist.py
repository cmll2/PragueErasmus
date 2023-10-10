
from io import TextIOWrapper
import numpy as np

def readInt(file : TextIOWrapper, byteorder : str) -> int:
    """Reads a 4-byte integer from the given file object using specific byte order."""
    b = file.read(4)
    return int.from_bytes(b, byteorder)

def loadMnist(imgFile : str, labelFile : str, readDigits : int = -1, offset : int = 0, trim : int = 0, normalize : bool = True) -> tuple:
    """
    Summary:
    Loads images and labels of the mnist digits dataset from files given as the first
    two arguments. Optionally reads only a number of images given by readDigits
    argument. It can also skip part of the dataset by specifying the offset argument.
    Lastly, the function can optionally trim the images using a given border width
    (argument trim) and normalise pixel values into the range 0-1.

    Arguments:
    - 'imgFile' - Path to the file with image data.
    - 'labelFile' - Path to the file with label data.
    - 'readDigits' - Number of images read from the file (-1 means all of them).
    - 'offset' - Number of images skipped at the beginning.
    - 'trim' - Number of pixels cut from each side of the image.
    - 'normalize' - Whether the values in images should be scaled to interval from 0 to 1.

    Returns:
    - An (N,K,K) array of images where K is 28-trim*2 for mnist images with side length 28 and
      N is the number of requested images.
    - An (N) array of integer labels for the requested images.
    """

    # Mnist is encoded using big endian.
    byteorder = "big"

    with open(imgFile, "rb") as f:

        # Check image header identifier.
        header = readInt(f, byteorder)
        if header != 2051:
            raise ValueError("Invalid image file header!")

        # Check the number of images in the file and their compatibility with
        # offset and ReadDigits values.
        count = readInt(f, byteorder)
        if count < offset:
            raise ValueError("Offset exceeds the data in the file!")
        if readDigits != -1 and count < readDigits + offset:
            raise ValueError("Trying to read too many digits!")

        height = readInt(f, byteorder)
        width = readInt(f, byteorder)

        # Skip part of the dataset.
        if offset > 0:
            f.seek(height * width * offset, 1)

        # Read the data.
        if readDigits != -1:
            data = np.fromfile(f, np.uint8, height * width * readDigits)
        else:
            data = np.fromfile(f, np.uint8)

    # Reshape the int array into a series of images.
    # The first axis enumerates the individual images.
    images = np.reshape(data, (-1, height, width))

    with open(labelFile, "rb") as f:

        # Check label header identifier.
        header = readInt(f, byteorder)
        if header != 2049:
            raise ValueError("Invalid image file header!")

        # Check the number of images in the file and their compatibility with
        # offset and ReadDigits values.
        count = readInt(f, byteorder)
        if count < offset:
            raise ValueError("Offset exceeds the data in the file!")
        if readDigits != -1 and count < readDigits + offset:
            raise ValueError("Trying to read too many digits!")

        # Skip part of the dataset.
        if offset > 0:
            f.seek(offset, 1)

        # Read the data.
        labels = np.fromfile(f, np.uint8, readDigits)

    # Optionally trim and normalise the images.
    if trim > 0:
        images = trimDigits(images, trim)
    if normalize:
        images = normalizePixelValues(images)

    return images, labels

def trimDigits(digits : np.ndarray, border : int) -> np.ndarray:
    """Trims images given in the stack 'digits' by cropping border of width specified by 'border'."""
    return digits[:, border : digits.shape[1] - border, border : digits.shape[2] - border]

def normalizePixelValues(digits : np.ndarray) -> np.ndarray:
    """Normalises the pixels values of 'digits' into the range 0-1."""
    return digits / 255.0
        