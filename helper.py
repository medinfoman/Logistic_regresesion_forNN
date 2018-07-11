import h5py
import numpy as np
from PIL import Image

def loaddata():
    f = h5py.File('data/trainingdata.hdf5', "r")
    train_x = np.array(f["train_x"][:])
    train_y = np.array(f["train_y"][:])
    
    f = h5py.File('data/testdata.hdf5', "r")
    test_x = np.array(f["test_x"][:])
    test_y = np.array(f["test_y"][:])
    
    return train_x, train_y, test_x, test_y

def flatten(train_x_raw, train_y_raw, test_x_raw, test_y_raw):
    train_flatten_x = train_x_raw.reshape(train_x_raw.shape[0], -1).T
    train_flatten_y = train_y_raw.reshape(train_y_raw.shape[0], -1).T
    test_flatten_x = test_x_raw.reshape(test_x_raw.shape[0], -1).T
    test_flatten_y = test_y_raw.reshape(test_y_raw.shape[0], -1).T
    
    train_set_x = train_flatten_x/255.
    test_set_x = test_flatten_x/255.
    train_set_y = train_flatten_y
    test_set_y = test_flatten_y
    
    return train_set_x, test_set_x, train_set_y, test_set_y

def image_arr(index, data_x):
    example = data_x[index]
    img = Image.fromarray(example, 'RGB')
    oneImage = np.array(example)
    
    return oneImage


def initialize_a_layer(layer, layer_r):
    np.random.seed(1)
    w = np.random.randn(layer, layer_r) / np.sqrt(layer)
    b = 0
    return w, b


def getRandomval(layer, layer_r):
    np.random.seed(1)
    w = np.random.randn(layer, layer_r)//1
    return w

def liner_function(w, X, b):
    z = np.dot(w.T, X) + b
    return z


def activation_function(z):
    s = 1. / ( 1 + np.exp(-z))
    return s

def getWrongImages(A, Y):
    prediction = np.array(A, copy=True)
    prediction[prediction<0.5]=0
    prediction[prediction>=0.5]=1

    FalsePositives = np.squeeze(np.absolute(prediction-Y))
    FalsePositives_arr = np.asarray(np.where(FalsePositives==1))
    
    num_img = len(np.squeeze(FalsePositives_arr))
    
    return np.squeeze(FalsePositives_arr)

def calculate_cost(Y, A):
    cost = (-1. / len(np.squeeze(Y))) * np.sum((Y*np.log(A) + (1 - Y)*np.log(1-A)), axis=1)
    return cost

def gradients(X, A, Y):
    numoffiles = len(np.squeeze(Y))
    
    dw = (1./numoffiles) * np.dot(X,((A-Y).T))
    db = (1./numoffiles) * np.sum(A-Y, axis=1)
    
    gradients = {"dw": dw, "db": db}
    
    return gradients


def prediction(w, b, X, Y):
    z = liner_function(w, X, b) # linear function
    A = activation_function(z) # activation function
    wrongImages = getWrongImages(A, Y)
    
    prediction = np.array(A, copy=True)
    prediction[prediction<0.5]=0
    prediction[prediction>=0.5]=1
    
    numoffile = float(Y.shape[1])
    numofwrong = float(len(wrongImages))
    
    accuracy = 1 - ( numofwrong / numoffile )
    
    result= {
        "wrongImages" : wrongImages,
        "accuracy" : accuracy,
        "A":A,
        "prediction":prediction,
    }
    
    return result


def train_model_(X, Y, test_X, test_Y, num_iterations, learning_rate):
    costs = []
    w, b = initialize_a_layer(12288, 1)
    
    for i in range(num_iterations):
        z = liner_function(w=w, X=X, b=b) # linear function
        A = activation_function(z) # activation function

        cost = calculate_cost(Y, A)

        if i%100==0:
            print(i, "th iteration cost: ", cost)
            costs.append(cost)

        grads = gradients(X, A, Y)

        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        
    
    # prediction
    result_train = prediction(w, b, X, Y)
    result_test = prediction(w, b, test_X, test_Y)
    

    print("training set accuracy: ", result_train["accuracy"])
    print("test set accuracy: ", result_test["accuracy"])
    
    result = {
        "result_train":result_train,
        "result_test":result_test,
        "costs":costs,
        "w":w,
        "b":b
    }
    
    return result