import random
import lzma
import cv2
import os
import pickle
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom
from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter
from functools import cache
from math import sqrt
from typing import *
from keras.datasets import mnist


def blur_image(image: np.ndarray, kernel_size_percentage: float) -> np.ndarray:
    """
    Blurs the given image by the specified percentage.

    Args:
    - image (numpy.ndarray): Input image as a numpy array.
    - kernel_size_percentage (float): Percentage of image width/height to use as the kernel size.

    Returns:
    - blurred_image (numpy.ndarray): Blurred image as a numpy array.
    """
    ## Calculate the kernel size
    sigma = round(1/4 * kernel_size_percentage * image.shape[0])
    return gaussian_filter(image, sigma=sigma)

def rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    """
    Rotates the given image by the specified number of degrees.
    """
   
    return rotate(image, degrees, reshape=False, order=1, mode='constant', cval=0.0, prefilter=False)


def shift_image(img: np.ndarray, horizontal_percent: float, vertical_percent: float) -> np.ndarray:
    """
    Shift the contents of an image by a specified percentage horizontally and vertically,
    padding with black/0. where new rows/columns are shifted in.

    Args:
    - img (numpy.ndarray): Input image as a numpy array.
    - horizontal_percent (float): Percentage of image width by which to shift horizontally.
    - vertical_percent (float): Percentage of image height by which to shift vertically.

    Returns:
    - shifted_img (numpy.ndarray): Shifted image as a numpy array.
    """
  
    return shift(img, (vertical_percent, horizontal_percent), order=1, mode='constant', cval=0.0, prefilter=False)

def zoom_image(img: np.ndarray, zoom_percentage: float) -> np.ndarray:
    zoomed =  zoom(img, zoom_percentage, order=1, mode='constant', cval=0.0, prefilter=False)
    # pad or crop to original dimensions
    if zoom_percentage > 1:
        # crop
        top = (zoomed.shape[0] - img.shape[0]) // 2
        bottom = zoomed.shape[0] - img.shape[0] - top
        left = (zoomed.shape[1] - img.shape[1]) // 2
        right = zoomed.shape[1] - img.shape[1] - left
        zoomed = zoomed[top:top + img.shape[0], left:left + img.shape[1]]
    else:
        # pad
        top = (img.shape[0] - zoomed.shape[0]) // 2
        bottom = img.shape[0] - zoomed.shape[0] - top
        left = (img.shape[1] - zoomed.shape[1]) // 2
        right = img.shape[1] - zoomed.shape[1] - left
        zoomed = cv2.copyMakeBorder(zoomed, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return zoomed


def adjust_contrast(image: np.ndarray, percentage:float) -> np.ndarray:
    """
    Adjust the contrast of an image using cv2.

    Args:
        image (np.ndarray): Input image as a numpy array.
        percentage (float): Contrast percentage. Positive values increase contrast, negative values decrease contrast.

    Returns:
        np.ndarray: Output image with contrast adjusted.
    """
    factor = (259 * (percentage + 255)) / (255 * (259 - percentage))
    adjusted = np.clip(128 + factor * image - factor * 128, 0, 255).astype(np.uint8)
    return adjusted

def generate_variations(images: np.ndarray[np.ndarray], labels: np.ndarray[np.ndarray], 
                        rotation_range: tuple, shift_range: tuple) -> tuple[np.ndarray, np.ndarray]:
    # determine new dimension (enlarge to 1.5 size for rotation and shifting without cropping)
    orig_dim = images[0].shape[0]
    new_dim_raw = int(orig_dim * 1.5)
    factor = 1
    new_dim = orig_dim
    # determine smallest integer multiple of the original dimension 
    # larger than or equal to 1.5x the original size to avoid artifacts
    while new_dim < new_dim_raw:
        factor += 1
        new_dim = int(orig_dim * factor)
    
    padding = (new_dim - orig_dim) // 2


    # replace every image by a randomly rotated and shifted version
    variations_images = np.empty_like(images)
    for x, img in enumerate(images):
        rotation_angle = np.random.uniform(rotation_range[0], rotation_range[1])

        shift_x = random.randint(shift_range[0], shift_range[1])
        shift_y = random.randint(shift_range[0], shift_range[1])

        new_img = np.zeros((new_dim, new_dim))


        new_img[padding:padding+img.shape[0], padding:padding+img.shape[0]] = img
        
        #rotate
        new_img = rotate(new_img, rotation_angle, order=1, mode='constant', cval=0.0, prefilter=False)

        # blur
        blur_size = random.randint(0, 3)
        if (blur_size > 0):
            new_img = blur_image(new_img, blur_size / 28)

        # take original image part
        new_img = new_img[padding:padding+img.shape[0], padding:padding+img.shape[0]]
        variations_images[x] = new_img
    
    print(f'Shape is {variations_images.shape}')

    return variations_images, labels

class ActivationFunction:
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        pass
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        pass
    def __call__(self, xs: np.ndarray[float]) -> np.ndarray[float]:
        return self.calculate(xs)

class Sigmoid(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return 1 / (1 + np.exp(-xs))
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        sig = Sigmoid.calculate(xs)
        return np.multiply(sig,(np.ones(len(xs)) - Sigmoid.calculate(xs)))

class Tanh(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.tanh(xs)
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return 1 - np.square(np.tanh(xs))

class ReLU(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.maximum(0, xs)
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return 1. * (xs > 0)

class LeakyReLU(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.maximum(0.01 * xs, xs)
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return 1. * (xs > 0) + 0.01 * (xs <= 0)

class SeLU(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(xs > 0, xs, 1.6732632423543772848170429916717 * (np.exp(xs) - 1))
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(xs > 0, np.ones(len(xs)), 1.6732632423543772848170429916717 * np.exp(xs))

class ELU(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(xs > 0, xs, 1.0 * (np.exp(xs) - 1))
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(xs > 0, np.ones(len(xs)), np.exp(xs))

class Softmax(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        exps = np.exp(xs - np.max(xs))
        return exps / np.sum(exps)
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return Softmax.calculate(xs) * (1 - Softmax.calculate(xs))

class Identity(ActivationFunction):
    @staticmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        return xs
    @staticmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        return np.ones(len(xs))


class LossFunction:
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        pass
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        pass
    @staticmethod
    def vectorized(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        orig_shape = expected.shape
        expected = expected.flatten()
        actual = actual.flatten()
        for i in range(len(expected)):
            expected[i] = LossFunction.derivative(expected[i], actual[i])
        result = expected.reshape(orig_shape)
        return result
    
    def vectorized_derivative(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        orig_shape = expected.shape
        expected = expected.flatten()
        actual = actual.flatten()
        for i in range(len(expected)):
            expected[i] = self.derivative(expected[i], actual[i])
        result = expected.reshape(orig_shape)
        return result

class MeanSquaredError(LossFunction):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum((expected - actual) ** 2) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return 2 * (actual - expected) / len(expected)

class MeanSquaredLogError(LossFunction):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum((np.log(expected + 1) - np.log(actual + 1)) ** 2) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return 2 * (np.log(actual + 1) - np.log(expected + 1)) / len(expected)

class MeanAbsoluteError(LossFunction):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum(np.abs(expected - actual)) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return np.sign(actual - expected) / len(expected)

class BinaryCrossEntropy(LossFunction):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return -np.sum(expected * np.log(actual) + (1 - expected) * np.log(1 - actual)) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return (actual - expected) / (actual * (1 - actual))

class HingeLoss(LossFunction):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum(np.maximum(0, 1 - expected * actual)) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return -expected * (1 - expected * actual > 0) / len(expected)

class SquaredHingeLoss(LossFunction):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum(np.maximum(0, 1 - expected * actual) ** 2) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return -2 * expected * (1 - expected * actual > 0) * (1 - expected * actual) / len(expected)

class CategoricalCrossEntropy(LossFunction):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        # Avoid taking the log of 0
        epsilon = 1e-15
        # Clip the actual values to avoid taking the log of values close to 0
        actual = np.clip(actual, epsilon, 1 - epsilon)
        return -np.sum(expected * np.log(actual))
    
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        # Avoid taking the log of 0
        epsilon = 1e-15
        # Clip the actual values to avoid taking the log of values close to 0
        actual = np.clip(actual, epsilon, 1 - epsilon)
        return actual - expected


class SparseCategoricalCrossEntropy(LossFunction):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return -np.sum(np.log(actual[expected == 1])) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return -expected / actual / len(expected)

class KullbackLeiblerDivergence(LossFunction):
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        return np.sum(expected * np.log(expected / actual)) / len(expected)
    @staticmethod
    def derivative(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return (np.log(expected / actual) - 1) / len(expected)


ArrayShape = TypeVarTuple('ArrayShape')
class ShapedData:
    def __init__(self, data: np.ndarray, shape: Tuple[*ArrayShape]):
        if type(data) != np.ndarray:
            raise Exception('Data must be a numpy array. Is ' + str(type(data)) + '.')
        self.data = data
        self.shape = shape
    def split_training_test(self, training_ratio: float) -> Tuple['ShapedData', 'ShapedData']:
        training_size = int(self.shape * training_ratio)
        training_data = self.data[:training_size]
        test_data = self.data[training_size:]
        return ShapedData(training_data, training_data.shape), ShapedData(test_data, test_data.shape)
    def split_training_validation_test(self, percentages: Tuple[float, float, float]) -> Tuple['ShapedData', 'ShapedData', 'ShapedData']:
        assert sum(percentages) == 1.0
        print(f'Splitting data into {percentages[0] * 100}% training, {percentages[1] * 100}% validation, and {percentages[2] * 100}% test.')
        data_len = len(self.data)
        training_size = int(data_len * percentages[0])
        validation_size = int(data_len * percentages[1])
        test_size = int(data_len * percentages[2])
        training_data = self.data[:training_size]
        validation_data = self.data[training_size:training_size + validation_size]
        test_data = self.data[training_size + validation_size:]
        return ShapedData(training_data, training_data.shape), ShapedData(validation_data, validation_data.shape), ShapedData(test_data, test_data.shape)



def zip_inputs_with_labels(inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # inputs is of shape (n, w, h), labels is of shape (n)
    # we need to create a new array of shape (n) where each element is a tuple of (input, label) with shape [(w, h), 1]
    output = np.empty(inputs.shape[0], dtype=object)
    for i in range(labels.shape[0]):
        value = [inputs[i], labels[i]]
        output[i] = value
    print(f'Data-set with length {len(output)} created.')
    return output

def combine_labeled_training_and_test_data(training_data: np.ndarray, test_data: np.ndarray) -> Tuple[ShapedData, float]:
    ratio = training_data.shape[0] / (training_data.shape[0] + test_data.shape[0])
    shape = training_data.shape[0] + test_data.shape[0]
    combined_data = np.concatenate((training_data, test_data))
    return ShapedData(combined_data, shape), ratio


def get_shaped_data_with_ratio_for_mnist() -> Tuple[ShapedData, Tuple[float, float, float]]:

    #see if there is a file at './data/mnist_transformed.p'. If so, load `shaped_data` from it.
    shaped_data = None
    if os.path.isfile('./data/mnist_transformed.p'):
        print('Loading data from file...')
        with lzma.open('./data/mnist_transformed.p', 'rb') as f:
            stored_data = pickle.load(f)
            shaped_data = stored_data['shaped_data']
            training_ratio, validation_ratio, test_ratio = stored_data['ratios']
    else:
        (train_inputs, train_labels), (test_inputs, test_labels) = mnist.load_data()
        # one-hot encode labels
        train_labels = np.array([np.eye(10)[x] for x in train_labels])
        test_labels = np.array([np.eye(10)[x] for x in test_labels])

        # augment images
        #print(f'Augmenting test images...')
        #test_inputs, test_labels = generate_variations(test_inputs, test_labels, (-30, 30), (-3, 3))
        #print(f'Shape of test_inputs after augmentation is {test_inputs.shape}')
        #print(f'Augmenting training images...')
        #train_inputs, train_labels = generate_variations(train_inputs, train_labels, (-30, 30),  (-3, 3))
        #print(f'Shape of train_inputs after augmentation is {train_inputs.shape}')
        # flatten inputs        
        train_inputs = train_inputs.reshape(train_inputs.shape[0], -1).astype('float32') / 255
        print(f'First example image: {train_inputs[0]}')
        test_inputs = test_inputs.reshape(test_inputs.shape[0], -1).astype('float32') / 255
        train_data_with_labels = zip_inputs_with_labels(train_inputs, train_labels)
        test_data_with_labels = zip_inputs_with_labels(test_inputs, test_labels)
        shaped_data, ratio = combine_labeled_training_and_test_data(train_data_with_labels, test_data_with_labels)
        # turn training/test split into training/validation/test split, taking 10% of the total data in proportial parts for training and test data
        # first, determine the number of data-points in training and test data based on ratio and length
        training_data_length = int(ratio * shaped_data.shape)
        test_data_length = shaped_data.shape - training_data_length
        # second, calculate how many data points are ten percent of total
        ten_percent = int(0.1 * shaped_data.shape)
        # third, split this number by ratio into the number to take from training and test data
        no_to_take_from_training = int(ratio * ten_percent)
        no_to_take_from_test = ten_percent - no_to_take_from_training
        # fourth, determine the new percentages of training and test data
        training_ratio = (training_data_length - no_to_take_from_training) / shaped_data.shape
        test_ratio = (test_data_length - no_to_take_from_test) / shaped_data.shape
        # round to two decimal places and ensure that the sum of the ratios is 1
        training_ratio = round(training_ratio, 2)
        test_ratio = round(test_ratio, 2)
        validation_ratio = 1 - training_ratio - test_ratio

        # pickle and store shaped_data with ratios
        data_to_store = {
            'shaped_data': shaped_data,
            'ratios': (training_ratio, validation_ratio, test_ratio)
        }
        with lzma.open('./data/mnist_transformed.p', 'wb') as f:
            pickle.dump(data_to_store, f)


    ## take one third of the data
    #shaped_data_third_arr = shaped_data.data[:int(shaped_data.shape / 3)]
    #shaped_data_third = ShapedData(shaped_data_third_arr, shaped_data_third_arr.shape)
    # display 10 random images from the training set using cv2
    # get the image source array from test data in shaped_data
    img_src = shaped_data.data
    print('Displaying 10 example images. Please press [ESC] on each to proceed.')
    for i in range(10):
        img_idx = np.random.randint(0, img_src.shape[0])
        img_tuple = img_src[img_idx]
        img = img_tuple[0]
        label_vect = img_tuple[1]
        label = np.argmax(label_vect)
        #set all values to be ints 
        img = (img * 255).astype('uint8').reshape(28, 28)

        cv2.imshow(f'image ({label})', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return (shaped_data, (training_ratio, validation_ratio, test_ratio))

def store_network(directory: str, filename: str) -> None:
    # check that directory exists and is writable
    if not os.path.isdir(directory):
        raise ValueError(f'Cannot store network in {directory} because it is not a directory')
    if not os.access(directory, os.W_OK):
        raise ValueError(f'Cannot store network in {directory} because it is not writable')
    # put weights and activation functions into a data structure for storage
    network = {
        'weights': weights,
        'activation_functions': ACTIVATION_FUNCTIONS,
    }
    # store network
    with open(os.path.join(directory, f'{filename}.pickle'), 'wb') as f:    
        pickle.dump(network, f)

def load_network(path: str) -> dict:
    # check that path exists and is readable
    if not os.path.isfile(path):
        raise ValueError(f'Cannot load network from {path} because it is not a file')
    if not os.access(path, os.R_OK):
        raise ValueError(f'Cannot load network from {path} because it is not readable')
    # load network
    with open(path, 'rb') as f:
        network = pickle.load(f)
    # check that network is valid
    if not isinstance(network, dict):
        raise ValueError(f'Cannot load network from {path} because it is not a dictionary')
    if 'weights' not in network:
        raise ValueError(f'Cannot load network from {path} because it does not contain weights')
    if 'activation_functions' not in network:
        raise ValueError(f'Cannot load network from {path} because it does not contain activation functions')
    return network


if __name__ == '__main__':
    # One neuron in each layer except the last is for the bias trick
    LAYER_SIZES = [785, 256, 128, 64, 10]
    ACTIVATION_FUNCTIONS = [Identity(), LeakyReLU(), LeakyReLU(), LeakyReLU(), Softmax()]
    WEIGHTS_INITIALIZER = lambda n, m: (np.random.randn(n*m) * sqrt(2.0 / n)).reshape(n, m)
    LOSS = CategoricalCrossEntropy()
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.0001
    MOMENTUM = 0.9
    EPOCHS = 10
    BATCH_SIZE = 256
    
    
    weight_decay_factor = 1 - LEARNING_RATE * WEIGHT_DECAY
    weights: np.ndarray = [None] + [WEIGHTS_INITIALIZER(n, m) for n, m in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])]
    loss_function = lambda y_true, y_pred: LOSS.calculate(y_true, y_pred)
    loss_derivative = lambda y_true, y_pred: LOSS.derivative(y_true, y_pred)
    
    data, ratio = get_shaped_data_with_ratio_for_mnist()    
    training, validation, test = data.split_training_validation_test(ratio)
    
    activations = [np.zeros((1, n)).astype("float32") for n in LAYER_SIZES]
    
    weighted_inputs = [None] + [np.zeros((n, m)).astype("float32") for n, m in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])]
    delta_weighted_inputs = [None] + [np.zeros((n, m)).astype("float32") for n, m in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])]
    
    # make each length n vector into a n+1 vector with the last element being 1 (bias trick)
    training_data = np.array([(np.append(x[0], 1), x[1]) for x in training.data], dtype=object)
    validation_data = np.array([(np.append(x[0], 1), x[1]) for x in validation.data], dtype=object)
    test_data = np.array([(np.append(x[0], 1),x[1]) for x in test.data], dtype=object)
    training_len = training_data.shape[0]
    validation_len = validation_data.shape[0]
    no_of_batches = int(training_len / BATCH_SIZE) + 1
    last_layer_idx = len(LAYER_SIZES) - 1
    no_of_classes = LAYER_SIZES[last_layer_idx]

    def fwd_pass(input):
        activations[0] = input
        for i in range(1, len(LAYER_SIZES)):
            z = activations[i - 1] @ weights[i]
            dZ = ACTIVATION_FUNCTIONS[i].derivative(z)
            weighted_inputs[i] = z
            delta_weighted_inputs[i] = dZ
            activations[i] = ACTIVATION_FUNCTIONS[i].calculate(z)
        return activations[-1]
    
    @cache
    def compile() -> Callable[[np.ndarray[float]], np.ndarray[float]]:
        def compiled(input_activation: np.ndarray[float]) -> np.ndarray[float]:
            curr_activations = input_activation
            for i in range(1, len(LAYER_SIZES)):
                curr_activations = ACTIVATION_FUNCTIONS[i].calculate(curr_activations @ weights[i])
            return curr_activations
        return compiled
    
    def evaluate(data_source, stage_name):   
        compiled = compile()     
        evaluation_loss = 0.
        confusion_matrix = np.zeros((no_of_classes, no_of_classes))
        count_per_class = np.zeros(no_of_classes)
        true_positives, true_negatives, false_positives, false_negatives = np.zeros(no_of_classes), np.zeros(no_of_classes), np.zeros(no_of_classes), np.zeros(no_of_classes)
        for (network_input, true_output) in data_source:
            predicted_output = compiled(network_input)
            prediction_loss = loss_function(true_output, predicted_output)
            evaluation_loss += prediction_loss
            if no_of_classes > 1:
                true_index = np.argmax(true_output)
                predicted_index = np.argmax(predicted_output)
                count_per_class[true_index] += 1
                confusion_matrix[true_index, predicted_index] += 1
                for prediction_class in range(no_of_classes):
                    is_true_class = (prediction_class == true_index)
                    is_pred_class = (prediction_class == predicted_index)
                    true_positives[prediction_class] += is_true_class and is_pred_class
                    false_negatives[prediction_class] += is_true_class and not is_pred_class
                    false_positives[prediction_class] += not is_true_class and is_pred_class
                    true_negatives[prediction_class] += not is_true_class and not is_pred_class
            else:
                true_index = 0 if true_output > 0.5 else 1
                predicted_index = 0 if predicted_output > 0.5 else 1
                confusion_matrix[true_index, predicted_index] += 1
                true_positives[0] += (predicted_output > 0.5) and (true_output > 0.5)
                false_positives[0] += (predicted_output > 0.5) and (true_output < 0.5)
                false_negatives[0] += (predicted_output < 0.5) and (true_output > 0.5)
                true_negatives[0] += (predicted_output < 0.5) and (true_output < 0.5)

        avg_evaluation_loss = evaluation_loss / len(data_source)

        accuracy_per_class = [true_positives[c] / (true_positives[c] + false_negatives[c]) if true_positives[c] + false_negatives[c] > 0 else 0. for c in range(no_of_classes)]
        precision_per_class = [true_positives[c] / (true_positives[c] + false_positives[c]) if true_positives[c] + false_positives[c] > 0 else 0. for c in range(no_of_classes)]
        recall_per_class = [true_positives[c] / (true_positives[c] + false_negatives[c]) if true_positives[c] + false_negatives[c] > 0 else 0. for c in range(no_of_classes)]
        f1_score_per_class = [2* p * r / (p + r) if (p + r) > 0 else 0. for p, r in zip(precision_per_class, recall_per_class)]
        
        accuracy_per_class_formatted = ['{:.4f}'.format(x) for x in accuracy_per_class]
        precision_per_class_formatted = ['{:.4f}'.format(x) for x in precision_per_class]
        recall_per_class_formatted = ['{:.4f}'.format(x) for x in recall_per_class]
        f1_score_per_class_formatted = ['{:.4f}'.format(x) for x in f1_score_per_class]

        avg_accuracy = np.mean(accuracy_per_class)
        avg_precision = np.mean(precision_per_class)
        avg_recall = np.mean(recall_per_class)
        avg_f1_score = np.mean(f1_score_per_class)

        print(f'{stage_name} loss: {evaluation_loss:.4f}[sum], {avg_evaluation_loss:.4f}[avg]')
        print(f'{stage_name} avg. accuracy: {avg_accuracy:.4f} - precision: {avg_precision:.4f} - recall: {avg_recall:.4f} - f1: {avg_f1_score:.4f}')
        print(f'{stage_name} accuracy per class: {accuracy_per_class_formatted}')
        print(f'{stage_name} precision per class: {precision_per_class_formatted}')
        print(f'{stage_name} recall per class: {recall_per_class_formatted}')
        print(f'{stage_name} f1 score per class: {f1_score_per_class_formatted}')


    no_of_layers_to_backpropagate = len(LAYER_SIZES) - 2
    N = 0
    for i in range(EPOCHS):
        print(f'Epoch {i + 1}')
        # shuffle training data
        np.random.shuffle(training_data)
        # split into batches
        i = 0
        start_idx = 0
        size_to_take = min(BATCH_SIZE, training_len - start_idx)
        curr_batch = training_data[start_idx:start_idx + size_to_take]
        velocity_term: np.ndarray = [None] + [np.zeros((n, m)) for n, m in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])]
        for b in range(no_of_batches):
            batch_loss = 0.
            batch_accuracy = 0.
            avg_batch_loss = 0.
            avg_batch_accuracy = 0.
            
            delta_incoming_weights_by_layer: np.ndarray  = [None] + [np.zeros((n, m)) for n, m in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])]
            # iterate over data points in batch
            t_p, t_n, f_p, f_n = np.zeros(no_of_classes), np.zeros(no_of_classes), np.zeros(no_of_classes), np.zeros(no_of_classes)
            curr_batch_size: int = curr_batch.shape[0]
            for x, y_true in curr_batch:
                size_per_class = np.zeros(no_of_classes)
                # do forward pass and update activations
                y_pred = fwd_pass(x)
                # do backward pass and update weights
                # calculate error
                loss = loss_function(y_true, y_pred)
                #print(f'evaluated loss fn on {y_true} and {y_pred} and got {loss}')
                if no_of_classes > 1:
                    y_true_idx = np.argmax(y_true)
                    y_pred_idx = np.argmax(y_pred)
                    for c in range(no_of_classes):
                        is_true = (c == y_true_idx)
                        is_pred = (c == y_pred_idx)
                        size_per_class[c] += is_true
                        t_p[c] += is_true and is_pred
                        f_n[c] += is_true and not is_pred
                        f_p[c] += not is_true and is_pred
                        t_n[c] += not is_true and not is_pred
                else:
                    t_p[0] += (y_pred > 0.5) and (y_true > 0.5)
                    f_p[0] += (y_pred > 0.5) and (y_true < 0.5)
                    f_n[0] += (y_pred < 0.5) and (y_true > 0.5)
                    t_n[0] += (y_pred < 0.5) and (y_true < 0.5)


                error = loss_derivative(y_true, y_pred).astype("float32")
                batch_loss += loss
                # weights-delta for last layer 
                delta = error * delta_weighted_inputs[-1]
                delta_incoming_weights_by_layer[-1] += np.outer(delta, activations[-2]).T
                # iterate backwards from last_layer_idx-1 to 1
                for l in range(no_of_layers_to_backpropagate, 0, -1):
                    delta = np.dot(weights[l+1], delta) * delta_weighted_inputs[l]
                    delta_incoming_weights_by_layer[l] += np.outer(delta, activations[l-1]).T
                N += 1
                    
            # update weights
            for l in range(1, no_of_layers_to_backpropagate + 1):
                grad = delta_incoming_weights_by_layer[l] / curr_batch_size + WEIGHT_DECAY * weights[l]
                velocity_term[l] = MOMENTUM*velocity_term[l] - LEARNING_RATE*grad
                weights[l] *= weight_decay_factor
                weights[l] += velocity_term[l]


            avg_batch_loss = batch_loss / curr_batch_size
            acc_per_class = [t_p[c] / (t_p[c] + f_n[c]) if t_p[c] + f_n[c] > 0 else 0. for c in range(no_of_classes)]
            precision_per_class = [t_p[c] / (t_p[c] + f_p[c]) if t_p[c] + f_p[c] > 0 else 0. for c in range(no_of_classes)]
            recall_per_class = [t_p[c] / (t_p[c] + f_n[c]) if t_p[c] + f_n[c] > 0 else 0. for c in range(no_of_classes)]
            f1_score_per_class = [2* p * r / (p + r) if (p + r) > 0 else 0. for p, r in zip(precision_per_class, recall_per_class)]
            avg_accuracy = np.mean(acc_per_class)
            avg_precision = np.mean(precision_per_class)
            avg_recall = np.mean(recall_per_class)
            avg_f1_score = np.mean(f1_score_per_class)

            print(f'Batch loss: {batch_loss:.4f}[sum], {avg_batch_loss:.4f}[avg] - avg. accuracy: {avg_accuracy:.4f} - avg. precision: {avg_precision:.4f} - avg. recall: {avg_recall:.4f} - avg. f1 score: {avg_f1_score:.4f}')
            # set for next iteration
            start_idx += size_to_take
            size_to_take = min(BATCH_SIZE, training_len - start_idx)
            curr_batch = training_data[start_idx:start_idx + size_to_take]
        # evaluate on validation data, unless we are on the last epoch
        if (i + 1) < EPOCHS:
            evaluate(validation_data, 'Validation')
    # evaluate on test data
    evaluate(test_data, 'Test')
    # store network
    store_network('./networks', 'mnist_256_256_128')

    