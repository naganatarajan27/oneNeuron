import numpy as np;
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_str)

class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4 # SMALL WEIGHT INIT
    logging.info("Initial weights before training: \n {0}".format(self.weights))
    self.eta= eta
    self.epochs=epochs
    
  def activationFunction(self, inputs, weights):
    z = np.dot(inputs,weights)
    return np.where(z > 0, 1, 0)

  def fit(self, X, y):  
    self.X = X
    self.y = y

    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))] # CONCATINATION
    logging.info("X_with_bias : \n {0}".format(X_with_bias))

    for epoch in range(self.epochs):
      logging.info("--"*10)
      logging.info("for epoch :  \n  {0}".format(epoch))
      logging.info("--"*10)
      y_hat = self.activationFunction(X_with_bias, self.weights) # forward propagation
      logging.info("predicted value after forward pass : \n {0}".format((y_hat)))
      self.error = self.y - y_hat
      logging.info("error :  \n {0}".format((self.error)))
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T,self.error) # backward propagation
      logging.info("updated weights after epoch: \n {0}/ {1} : {2}".format(epoch,self.epochs,self.weights))
      logging.info("#####"*10)


  def predict(self, X):
     X_with_bias = np.c_[X, -np.ones((len(X), 1))]
     return self.activationFunction(X_with_bias, self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    logging.info("total loss :  \n {}".format(total_loss))
    return total_loss

