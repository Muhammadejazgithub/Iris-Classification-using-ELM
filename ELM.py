
"""
Extreme Learning Machine Classification
"""

import numpy as np
from scipy.linalg import pinv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import r2_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import h5py
import pandas as pd


class _BaseELM:
    """Base class for ELM classification and regression."""

    def __init__(self, hidden_layer_neurons, activation, random_state=None):
        if hidden_layer_neurons < 1:
            raise ValueError("hiddenSize must be greater than 0")
        
        self.hidden_layer_neurons = hidden_layer_neurons
        self.activationName = activation
        self.activation = self._activationFunction(self.activationName)
        self.random_state = random_state
        self.isFitted = False


    def _identity(self, Z):
        
        return Z

    def _sigmoid(self, Z):

        
        # Avoid overflow
        Z = np.clip(Z, -709, 709)
        return 1 / (1 + np.exp(-Z))

    def _relu(self, Z):
        
        return np.maximum(Z, 0, Z)

    def _tanh(self, Z):
        
        return np.tanh(Z)

    def _activationFunction(self, functionName):

        
        if functionName == 'sigmoid':
            return self._sigmoid
        if functionName == 'relu':
            return self._relu
        if functionName == 'tanh':
            return self._tanh
        if functionName == 'identity':
            return self._identity
        else:
            raise ValueError("Not a valid activation function")

    def _hidden_layer_output(self, x):
        
        A = np.dot(x, self.weight) + self.bias
        A = self.activation(A)
        return A

    def _check_is_fitted(self):
        
        if self.isFitted:
            return self
        else:
            raise Exception(
                "This model is not fitted yet. Call 'fit' with appropriate arguments before using this model.")

    def _fit(self, X, y):
        
        X = np.array(X)
        y = np.array(y)
        self.n_samples, self.n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        # Set the Random seed
        np.random.seed(self.random_state)
        # Initialize weights
        self.weight = np.random.normal(0,0.5,size=[self.n_features, self.hidden_layer_neurons])
        # Initialize bias
        self.bias = np.random.normal(0.,0.5,size=self.hidden_layer_neurons)
        # Calculate hidden layer output matrix (Hinit)
        H = self._hidden_layer_output(X)
        # Calculate the Moore-Penrose pseudoinverse matriks
        H_moore_penrose = pinv2(H)
        # Calculate the output weight matrix beta
        self.beta = np.dot(H_moore_penrose, y)
        self.isFitted = True
        return self

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.
        """
        
        return self._fit(X, y)

    def _predict(self, X):
        
        """Predict using the trained model
        """
        y_pred = self._hidden_layer_output(X)
        y_pred = np.dot(y_pred, self.beta)
        return y_pred

    def get_params(self):
        
        """
            Get parameters for this model.
            Returns
            -------
            params : mapping of string to any
                Parameter names mapped to their values.
            """
        out = {
            'hidden_layer_neurons': self.hidden_layer_neurons,
            'activation': self.activationName,
            'random_state': self.random_state
        }
        return out

class ELMCLassifier(_BaseELM):
    """Extreme Learning machine classifier.
    """

    def __init__(self, hidden_layer_neurons, activation, random_state=None):
        super().__init__(hidden_layer_neurons=hidden_layer_neurons, activation=activation, random_state=random_state)
        self._label_binarizer = LabelBinarizer()

    def fit(self, X, y):
        
        """Fit the model to data matrix X and target(s) y.
        self : returns a trained ELM model.
        """

        y = self._label_binarizer.fit_transform(y)
        return self._fit(X, y)

    def predict(self, X):
        
        """Predict using the extreme learning machine classifier
        """
        self._check_is_fitted()
        y_pred = self._predict(X)

        return y_pred, self._label_binarizer.inverse_transform(y_pred)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        """
        
        
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


#argument parsing
def pars_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("hidden_layer_neurons",help="Input hidden layer neurons",type=int,default=50,nargs='?')
    parser.add_argument("activation",help="Input activation function name (tanh, relu, sigmoid, identity)",default='sigmoid',nargs='?')

    args = parser.parse_args()

    return args


#### main function ######
def main():
    
    classes=["Iris-setosa","Iris-versicolor","Iris-virginica",]
    args= pars_arguments()
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    elmc = ELMCLassifier(hidden_layer_neurons=args.hidden_layer_neurons, activation=args.activation)
    model =elmc.fit(X_train, y_train)

    inputvalues= [[4.4,3.2,1.3,0.2],[6.2,2.8,4.8,1.8],[4.9,3.1,1.5,0.2],[5.5,2.3,4.0,1.3],[6.3,2.9,5.6,1.8],[6.3,2.5,4.9,1.5],
    [5.8,2.7,4.1,1.0],[4.9,2.4,3.3,1.0],[5.6,3.0,4.5,1.5],[6.0,2.9,4.5,1.5]]

    #print("List of Input values",inputvalues)
    proba, predictions = model.predict(inputvalues)

    classone=[]
    classtwo=[]
    classthree=[]
    maxprob=[]

    for i in range(len(proba)):
        classone.append(format(proba[i][0],".3f"))
        classtwo.append(format(proba[i][1],".3f"))
        classthree.append(format(proba[i][2],".3f"))
        maxprob.append(format(np.max(proba[i]),".3f"))

    print("Input values: ",inputvalues)
    print("classone: ",classone)
    print("classtwo: ",classtwo)
    print("classthree: ",classthree)
    #print("Max probabilities: ",maxprob)
    print("Class with highest probability:",predictions)
    
    dict = {'Input Values(sepal_lenght,width, petal_length,width)': inputvalues, 'probability of class 0 (setosa)': classone, 'probability of class 1 (versicolor)': classtwo,'probability of class 2 (virginica)': classthree, 'Class with highest probability': predictions} 
    df = pd.DataFrame(dict) 
    df.to_csv('sigmoid_with_normal_a=0_b=0.5_neuron50.csv')


    #print('Model accuracy on test values: %f' % elmc.score(X_test, y_test))


if __name__ == '__main__':
    main()