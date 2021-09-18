import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib # FOR SAVING MY MODEL AS A BINARY FILE
from matplotlib.colors import ListedColormap
import os

plt.style.use("fivethirtyeight") # THIS IS STYLE

def prepare_data(df):
  """It is used to seperate the dependent and independent features
  Args:
      df (pd DataFrame): its the pandas Dataframe to

  Returns:
      tuple: it return the tuples of dependent variables and independent variables
  """
  X=df.drop("y", axis=1)
  y=df["y"]
  return X , y 

def save_model(model, filename):
  """It is used to save the trained model in binary format

  Args:
      model (python object): trained model
      filename (string): file name to save the trained model
  """
  model_dir = "models"
  os.makedirs(model_dir,exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN'T EXIST
  filepath=os.path.join(model_dir,filename) # model/filename
  joblib.dump(model,filepath)

def save_plot(df,file_name,model):

  """It is used to save the plot based on the trained model

    Args:
        df (pd DataFrame): its the Pandas DataFrame
        file_name (string ): file name to save the plot
        model (python object): trained model
    """

  def _create_base_plot(df):
    """It is used to create the plot graph

    Args:
        df (pd DataFrame): its the Pandas DataFrame
    """

    df.plot(kind = "scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0,color="black",linestyle="--",linewidth=1)
    plt.axvline(x=0,color="black",linestyle="--",linewidth=1)
    figure = plt.gcf()
    figure.set_size_inches(10,8)

  def _plot_decission_regions(X, y, classfier, resolution = 0.02):
    """It is used to create the plot based on the trained model Data

    Args:
        X (int array): [pd Dataframe 1st Object]
        y (int array): [pd Dataframe 2nd Object]
        classfier (python Object): Trained Model
        resolution (float, optional): The plot image Resolution. Defaults to 0.02.
    """
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors [:len(np.unique(y))])

    X = X.values # as a array

    x1_min, x1_max = X[:, 0].min()-1, X[:,0].max() +1
    x2_min, x2_max = X[:, 1].min()-1, X[:,1].max() +1

    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution), 
                           np.arange(x2_min,x2_max,resolution))
    
    Z = classfier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha =0.2, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    plt.plot


  X, y = prepare_data(df)

  _create_base_plot(df)

  _plot_decission_regions(X, y, model)

  plot_dir ="plots"
  os.makedirs(plot_dir,exist_ok=True)
  plotPath = os.path.join(plot_dir,file_name)
  plt.savefig(plotPath)

