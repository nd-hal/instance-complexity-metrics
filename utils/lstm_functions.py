import keras
from keras import Model
# from keras.layers import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.preprocessing import sequence
from keras.optimizers import Adam

# from utils.utility_functions import *


###################################################################
################### Create the Model ##############################
###################################################################
# we use the keras LSTM instead of tensorflow due to efficiency gains
def createWordLSTM(trainSet, numClasses, numLstmLayers, lstmNodes, learningRate,
                   keras_embs=False, to_output=True): 
    trainFeatures = trainSet.pt_embeddings
    input = Input(shape=trainFeatures.shape[1:], dtype='float32')
    
    # loop thru to add lstm layers
    if numLstmLayers > 1:
        lstm = Bidirectional(LSTM(lstmNodes, dropout=0.2, return_sequences=True))(input)
        for _ in range(numLstmLayers-1):
            ret_seq = False if _==numLstmLayers-2 else True
            lstm = Bidirectional(LSTM(lstmNodes, dropout=0.2, return_sequences=ret_seq))(lstm)
    else:
        lstm = Bidirectional(LSTM(lstmNodes, dropout=0.2, return_sequences=False))(input)

    dense = Dense(64, activation='relu')(lstm)

    # pass along to be concatenated
    if not to_output:
        return dense, input

    # finish off the model
    else:

        # output layer
        output = Dense(numClasses, activation="softmax")(dense)      
        model = Model(inputs=input, outputs=output)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(learning_rate=learningRate),
                      metrics=['accuracy'])
        return model
