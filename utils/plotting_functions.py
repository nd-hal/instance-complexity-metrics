import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###################################################################
################### Plotting a Confusion Matrix ###################
###################################################################
def addText(xticks, yticks, results):
    """Add text in the plot"""
    for i in range(2):
        for j in range(2):
            pltText = plt.text(j, i, results[i][j], ha="center", va="center", color="white", size=15) ### size here is the size of text inside a single box in the heatmap    

def displayConfusionMatrix(confusionMatrix):
    """Confusion matrix plot"""
    
    confusionMatrix = np.transpose(confusionMatrix)
    
    ## calculate class level precision and recall from confusion matrix
    precisionLow = round((confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[0][1]))*100, 1)
    precisionHigh = round((confusionMatrix[1][1] / (confusionMatrix[1][0] + confusionMatrix[1][1]))*100, 1)
    recallLow = round((confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]))*100, 1)
    recallHigh = round((confusionMatrix[1][1] / (confusionMatrix[0][1] + confusionMatrix[1][1]))*100, 1)

    ## show heatmap
    plt.imshow(confusionMatrix, interpolation='nearest',cmap=plt.cm.Blues,vmin=0, vmax=100)
    
    ## axis labeling
    PLOT_FONT_SIZE = 14
    xticks = np.array([-0.5,0,1,1.5])
    plt.gca().set_xticks(xticks)
    plt.gca().set_yticks(xticks)
    plt.gca().set_xticklabels(["","LowSev \n Recall=" + str(recallLow), "HighSev \n Recall=" + str(recallHigh),""], fontsize=PLOT_FONT_SIZE)
    plt.gca().set_yticklabels(["","LowSev \n Precision=" + str(precisionLow), "HighSev \n Precision=" + str(precisionHigh),""], fontsize=PLOT_FONT_SIZE)
    plt.ylabel("Predicted Class", fontsize=PLOT_FONT_SIZE)
    plt.xlabel("Actual Class", fontsize=PLOT_FONT_SIZE)

    ## add text in heatmap boxes
    addText(xticks, xticks, confusionMatrix)