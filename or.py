import pandas as pd
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_str)

def main(data,eta,epochs,modelName,plotName):

    df = pd.DataFrame(data)
    logging.info(df)
    logging.info("This is the data frame {0}".format(df))
    X, y = prepare_data(df)
    model =Perceptron(eta=eta,epochs=epochs)
    model.fit(X,y)
    _ = model.total_loss()
    save_model(model,filename=modelName)
    save_plot(df,filename=plotName,model=model)

if __name__ == '__main__':
    OR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,1]
    }   
    ETA = 0.3 # 0 and 1
    EPOCHS=10
    
    try:
        main(OR, ETA, EPOCHS,"or.model","or.png")
    except Exception as e:
        logging.exception(e)