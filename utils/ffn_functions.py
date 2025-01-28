import keras
from keras import Model
from keras.layers import *

# from utils.utility_functions import *


###################################################################
################### Create the model ##############################
###################################################################
def createFFN(trainSet, numClasses, numLayers, layerNodes, learningRate, keras_embs=False): 
    """Create a feed forward neural network"""
    trainFeatures = trainSet.pt_embeddings
    vocabLen = len(trainSet.vocab_d)

    # create basic cnn model
    modelInput = Input(shape=trainFeatures.shape[1:], dtype='float32')
 
    ## word convolutional neural network
    if keras_embs:
        # Create embeddings using keras built in function.
        neuralNetworkLayer = keras.layers.Embedding(input_dim=vocabLen + 1, 
                                   output_dim=trainSet.vec_len, 
                                   input_length=len(trainFeatures[0]))(modelInput)
        for i in range(numLayers - 1):
            neuralNetworkLayer = keras.layers.Dense(layerNodes, activation='relu')(neuralNetworkLayer)
            neuralNetworkLayer = keras.layers.Dropout(0.5)(neuralNetworkLayer)
    else:
        
        # Here, we are using pre-trained embeddings. Therefore, we don't need to call layers.embeddings function.
        neuralNetworkLayer = keras.layers.Dense(layerNodes, activation='relu')(modelInput)
        neuralNetworkLayer = keras.layers.Dropout(0.5)(neuralNetworkLayer)
        for i in range(numLayers - 1):
            neuralNetworkLayer = keras.layers.Dense(layerNodes, activation='relu')(neuralNetworkLayer)
            neuralNetworkLayer = keras.layers.Dropout(0.5)(neuralNetworkLayer)

    
    # You can change the number of nodes in the dense layer. Right now, it's set to 32.
    denseLayer = keras.layers.Dense(32)(neuralNetworkLayer)
    flatLayer = keras.layers.Flatten()(denseLayer)

    output = keras.layers.Dense(numClasses, activation='softmax')(flatLayer)
    model = Model(inputs=modelInput, outputs=output)

    # NOTE; setting the optimizer outside in main
    # optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # model.compile(optimizer=optimizer,
    #                 loss='binary_crossentropy', 
    #                 metrics=['accuracy'])

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=learningRate),
                  metrics=['accuracy'])

    return model


