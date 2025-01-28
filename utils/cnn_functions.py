import keras
import keras
from keras import Model
from keras.layers import *

# from utils.utility_functions import *


###################################################################
################### Create the model ##############################
###################################################################
def createWordCNN(trainSet, numClasses, numLayers, learningRate, numFilters,
                  kernelSize, keras_embs=False, to_output=True): 
    """Create a word cnn"""
    trainFeatures = trainSet.pt_embeddings
    vocabLen = len(trainSet.vocab_d)

    ## create basic cnn model
    wordInput = Input(shape=trainFeatures.shape[1:], dtype='float32')
 
    ## word convolutional neural network
    if keras_embs:
        # Create embeddings using keras built in function.
        wordCNN = keras.layers.Embedding(input_dim=vocabLen + 1, 
                                   output_dim=trainSet.vec_len, 
                                   input_length=len(trainFeatures[0]))(wordInput)
    
        # Add CNN layers equal to numConvLayers
        for i in range(numLayers):
            wordCNN = keras.layers.Conv1D(numFilters, kernelSize, activation='relu')(wordCNN)
            wordCNN = keras.layers.Dropout(0.5)(wordCNN)
    else:
        
        # Here, we are using pre-trained embeddings. Therefore, we don't need to call layers.embeddings function.
        wordCNN = keras.layers.Conv1D(numFilters, kernelSize, activation='relu', input_shape=trainFeatures.shape[1:])(wordInput)
        wordCNN = keras.layers.Dropout(0.5)(wordCNN)
        for i in range(numLayers - 1):
            wordCNN = keras.layers.Conv1D(numFilters, kernelSize, activation='relu')(wordCNN)
            wordCNN = keras.layers.Dropout(0.5)(wordCNN)
    
    # GlobalMaxPooling is a good function to use for pooling operations, let's keep it like this
    wordCNN = keras.layers.GlobalMaxPooling1D()(wordCNN)
    wordCNN = keras.layers.Dropout(0.5)(wordCNN)
    
    # You can change the number of nodes in the dense layer. Right now, it's set to 64.
    denseLayer = keras.layers.Dense(64)(wordCNN)

    # we want to concatenate it, don't we folks
    if not to_output:
        return denseLayer, wordInput


    flatLayer = keras.layers.Flatten()(denseLayer)

    output = keras.layers.Dense(numClasses, activation='softmax')(flatLayer)
    model = Model(inputs=wordInput, outputs=output)

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learningRate),
                  metrics=['accuracy'])

    return model



# simlar model for character level
def createCharCNN(trainSet, numClasses, numLayers, learningRate, numFilters, kernelSize, keras_embs=False): 
    """Create a char cnn"""
    trainFeatures = trainSet.pt_embeddings
    
    ## create basic cnn model
    charInput = Input(shape=trainFeatures.shape[1:], dtype='float32')
 
    # Here, we are using one hot encoded representation of characters. Therefore, we don't need to call layers.embeddings function.
    charCNN = keras.layers.Conv1D(numFilters, kernelSize, activation='relu', input_shape=trainFeatures.shape[1:])(charInput)
    charCNN = keras.layers.Dropout(0.2)(charCNN)
    for i in range(numLayers - 1):
        charCNN = keras.layers.Conv1D(numFilters, kernelSize, activation='relu')(charCNN)
        charCNN = keras.layers.Dropout(0.2)(charCNN)
    
    charCNN = keras.layers.GlobalMaxPooling1D()(charCNN)
    
    # You can change the number of nodes in the dense layer. Right now, it's set to 64.
    denseLayer = keras.layers.Dense(64)(charCNN)
    flatLayer = keras.layers.Flatten()(denseLayer)


    output = keras.layers.Dense(numClasses, activation='softmax')(flatLayer)
    model = Model(inputs=charInput, outputs=output)


    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=learningRate),
                  metrics=['accuracy'])
    
    return model








# def createWordCNN(trainSet, numClasses, numLayers, learningRate, numFilters, kernelSize, keras_embs=False): 
#     """Create a word cnn"""
#     trainFeatures = trainSet.pt_embeddings
    
#     # ## create basic cnn model
#     # wordInput = Input(shape=trainFeatures.shape[1:], dtype='float32')

#     if keras_embs:
#         trainVocab = trainSet.vocab_d
#         modelInput = Input(shape=trainFeatures.shape[1], dtype='float32')
#         modelInput = keras.layers.Embedding(input_dim=len(trainVocab)+1, 
#                                     output_dim=trainSet.vec_len, 
#                                     input_length=trainSet.max_len)(modelInput)
#     else:
#         # small change in .shape[1:]
#         modelInput = Input(shape=trainFeatures.shape[1:], dtype='float32')

 
#     wordCNN = keras.layers.Conv1D(numFilters, kernelSize, activation='relu', input_shape=trainFeatures.shape[1:])(modelInput)
#     wordCNN = keras.layers.Dropout(0.5)(wordCNN)
#     for i in range(numLayers - 1):
#         wordCNN = keras.layers.Conv1D(numFilters, kernelSize, activation='relu')(wordCNN)
#         wordCNN = keras.layers.Dropout(0.5)(wordCNN)
    
#     # GlobalMaxPooling is a good function to use for pooling operations, let's keep it like this
#     wordCNN = keras.layers.GlobalMaxPooling1D()(wordCNN)
#     wordCNN = keras.layers.Dropout(0.5)(wordCNN)
    
#     # You can change the number of nodes in the dense layer. Right now, it's set to 64.
#     denseLayer = keras.layers.Dense(64)(wordCNN)
#     flatLayer = keras.layers.Flatten()(denseLayer)

#     output = keras.layers.Dense(numClasses, activation='softmax')(flatLayer)
#     model = Model(inputs=modelInput, outputs=output)

#     # optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
#     # model.compile(optimizer=optimizer,
#     #                 loss='binary_crossentropy', 
#     #                 metrics=['accuracy'])

#     model.compile(loss=keras.losses.sparse_categorical_crossentropy,
#                   optimizer=keras.optimizers.Adam(lr=learningRate),
#                   metrics=['accuracy'])

#     return model
