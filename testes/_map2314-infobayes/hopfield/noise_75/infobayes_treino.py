import numpy as np

import matplotlib.pyplot as plt

import skimage.data
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
from skimage import img_as_bool

import network

def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted
    
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(5, 6)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    print(fig, axarr[0], len(data))
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Treino')
            axarr[i, 1].set_title("Teste")
            axarr[i, 2].set_title('Predição')

        axarr[i, 0].imshow(data[i], cmap='gray')
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i], cmap='gray')
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i], cmap='gray')
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

def preprocessing(img, w=100, h=100):
    # Resize image
    img = resize(img, (w,h), mode='reflect')

    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int

    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def main():
    # Load data
    shannon = io.imread('shannon.png')
    renato = io.imread('renato.png')
    boltzmann = io.imread('boltzmann.png')

    # Marge data
    data = [shannon, renato, boltzmann]

    # Preprocessing
    print("Start to data preprocessing...")
    data = [preprocessing(d) for d in data]

    # Create Hopfield Network Model
    model = network.HopfieldNetwork()
    model.train_weights(data)

    # Generate testset
    test = [get_corrupted_input(d, 0.75) for d in data]

    predicted = model.predict(test, threshold=0, asyn=True)
    print("Show prediction results...")
    plot(data, test, predicted)
    print("Show network weights matrix...")
    model.plot_weights()

if __name__ == '__main__':
    main()
