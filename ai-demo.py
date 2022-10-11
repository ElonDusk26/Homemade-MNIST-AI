from ai import NeuralNet #fully made by me without the use of external libraries 
from mnist import MNIST
import numpy as np
from PIL import Image


##define useful helper functions. 
def translateLabelsIntoNetwork(labelsList, dimensions):
    out = []
    for label in labelsList:
        lst = [0] * dimensions
        lst[label] = 1
        out.append(lst)
    return out

def scaleTrainingImages(images, scalar): #not needed to create a copy and it slows down execution, but you can preserve the original variable if its desired
    out = []
    for image in images:
        out.append(np.multiply(image, scalar))
    return out

def maxNumberIndex(numberList):
    return numberList.index(max(numberList))

def showImage(imageArr):
    image = Image.fromarray(np.uint8(np.array(imageArr).reshape((28,28))), mode='L')
    image.show()

def showNormalizedImage(imageArr):
    npImage = np.array(imageArr).reshape((28,28))
    scaledNpImage = np.multiply(npImage, 255)
    image = Image.fromarray(np.uint8(scaledNpImage), mode='L')
    image.show()


#Load the MNIST database
mnistData = MNIST()

trainingImages, trainingLabels = mnistData.load_training()

trainingImages = scaleTrainingImages(trainingImages, 1/255) #normalize pixels from intervals 0-255 to 0-1

testingImages, testingLabels = mnistData.load_testing()

testingImages = scaleTrainingImages(testingImages, 1/255) #normalize pixels from intervals 0-255 to 0-1

trainingLabels = translateLabelsIntoNetwork(trainingLabels, 10) #create an array based off of the index instead of a number. if the label is 7 then it gets translated into [0,0,0,0,0,0,0,1,0,0], where the seventh index is 1. 

testingLabels = translateLabelsIntoNetwork(testingLabels, 10) #repeat but for testing labels

nn = NeuralNet() #instantiate a neural net written by me using slow python list comprehensions. Made that way so i could learn about how an ai works. Not intended for speed.

nn.addLayer(784, 100) #input dimensions are 784 because thats the amount of pixels in an image

nn.addLayer(100, 10) #hidden layer of 100 nodes and an output layer of 10 nodes. 
##layers can be added as needed and wanted. The only limitation is hardware


#train the model on the chosen dataset. Ive chosen the first 2500 images of the training dataset.
#Then specify the testing images and labels, so its possible to understand the accuracy of the model in a general usecase. If it has a high accuracy when training but low when testing, then the AI is overfitted.
#the training method used here has a hardcoded noise function. It adds random noise to pixels with a lower value than 0.05. 
#The learning rate is then specified to 0.08, which is the coefficient used to scale down the changes from the derivatives found after the backpropagation phase.
#lastly the epoch is specified, which describes the amount of times to loop through the training data.
nn.learnFromDataset(trainingImages[0:500], trainingLabels[0:500], testingImages[0:100], testingLabels[0:100], 0.05, 1) 

#now the model is trained, and i would like to see the results myself.

correctClassifications = 0 
numberOfImages = 100
for image, label in zip(testingImages[0:numberOfImages], testingLabels[0:numberOfImages]):
    out = nn.computeNetwork(image) #compute the output made by the model
    if maxNumberIndex(label) == maxNumberIndex(out): #if the output by the model matches with the correct label then it counts towards the correctClassifications counter
        correctClassifications += 1
print("accuracy of {}%".format((correctClassifications/numberOfImages)*100)) #calculating the percentages


#here im learning with different training images to showcase the change in accuracy by training the same model again. This process does not reset the model, it adjusts the already slightly trained model.
nn.learnFromDataset(trainingImages[500:1000], trainingLabels[500:1000], testingImages[0:100], testingLabels[0:100], 0.05, 1) 


#exact same method. Go through the first 100 test images, which the AI has never seen before to get an accuracy rating.
#hopefully an increase in accuracy is achieved :)
correctClassifications = 0
numberOfImages = 100
for image, label in zip(testingImages[0:numberOfImages], testingLabels[0:numberOfImages]):
    out = nn.computeNetwork(image)
    if maxNumberIndex(label) == maxNumberIndex(out):
        correctClassifications += 1
print("accuracy of {}%".format((correctClassifications/numberOfImages)*100))