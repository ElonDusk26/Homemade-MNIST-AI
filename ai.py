from mnist import MNIST
import numpy as np
from PIL import Image
import random
import math
import threading

def showImage(imageArr):
    image = Image.fromarray(np.uint8(np.array(imageArr).reshape((28,28))), mode='L')
    image.show()

class NeuralNet():
    def __init__(self):
        self.layers = []

    def activationFunction(self, input):
        if -30 > input:
            return 0
        elif input > 30:
            return 1
        return (1/(1+math.pow(math.e,(-input))))

    def activationFunctionDerivative(self, input):
        return self.activationFunction(input) * (1-self.activationFunction(input))

    def useActivationOnList(self, listInp):
        out = []
        for number in listInp:
            out.append(self.activationFunction(number))
        return out

    def useActivationDerivativeOnList(self, listInp):
        out = []
        for number in listInp:
            out.append(self.activationFunctionDerivative(number))
        return out

    def addLayer(self,inputNodes, outputNodes):
        self.layers.append(self.Layer(inputNodes, outputNodes))

    def computeNetwork(self, input): #goes through the layers from start to finish with sigmoid(ax+b)
        output = self.layers[0].computeLayerValues(input) #no activation layer on the input. The input is "as is"
        for i in range(len(self.layers) - 1):
            output = self.useActivationOnList(self.layers[i+1].computeLayerValues(output))
        return output

    def lossFunction(self, input, expectedInput):
        return math.pow(input-expectedInput, 2)
        
    def lossFunctionDerivative(self, input, expectedInput):
        return 2*(input - expectedInput)

    def lossFunctionForList(self, listInp, expectedListInp): #just the error squared for each neuron.
        output = []
        for i in range(len(listInp)):
            output.append(self.lossFunction(listInp[i], expectedListInp[i]))
        return output

    def lossFunctionDerivativeForList(self, listInp, expectedListInp): #just the error squared for each neuron.
        output = []
        for i in range(len(listInp)):
            output.append(self.lossFunctionDerivative(listInp[i], expectedListInp[i]))
        return output



    def backpropagation(self, networkOutput, expectedOutput):
       #networkOutput = self.computeNetwork(input)

        '''
        outputCostDerivative = self.lossFunctionDerivativeForList(networkOutput, expectedOutput) #reusable
        outputActivationDerivative = self.useActivationDerivativeOnList(networkOutput) #reusable
        layerWeightDerivative = self.useActivationOnList(self.layers[-1-1].previousOutputValues) #reuses the activation function which isnt optimal. bad code.

        for i in range(self.layers[-1].outputLen):
            for j in range(self.layers[-1-1].outputLen):
                self.layers[-1].weightDerivatives[i][j] = outputCostDerivative[i] * outputActivationDerivative[i] * layerWeightDerivative[j]
                self.layers[-1].biasDerivatives[i] = outputCostDerivative[i] * outputActivationDerivative[i]

        '''



        outputCostDerivative = self.lossFunctionDerivativeForList(networkOutput, expectedOutput) #reusable
        outputActivationDerivative = self.useActivationDerivativeOnList(networkOutput) #reusable
        
        nodeValue = []
        for i in range(len(outputActivationDerivative)):
            nodeValue.append(outputActivationDerivative[i] * outputCostDerivative[i])
            
        for layerIndex in range(len(self.layers)):
            if layerIndex > 0:
                newNodeValue = [0] * self.layers[-1 - layerIndex].outputLen
                for i in range(len(newNodeValue)): #calculating z2 over a1
                    for j in range(len(nodeValue)):
                        newNodeValue[i] += nodeValue[j] * self.layers[0 - layerIndex].weights[j][i] 

                for i in range(len(newNodeValue)): #calculating a1 over z1
                    newNodeValue[i] *= self.activationFunctionDerivative(self.layers[-1-layerIndex].previousOutputValues[i])

                nodeValue = newNodeValue

            layerWeightDerivative = 0
            if layerIndex == len(self.layers)-1:
                layerWeightDerivative = self.layers[-1-(layerIndex)].previousInputValues #skip activation layer on last input since the input dosent have a func
            else:
                layerWeightDerivative = self.useActivationOnList(self.layers[-1-(layerIndex)].previousInputValues)

            for i in range(self.layers[-1-layerIndex].outputLen):
                self.layers[-1-layerIndex].biasDerivatives[i] = nodeValue[i]
                for j in range(self.layers[-1-layerIndex].inputLen):
                    self.layers[-1-layerIndex].weightDerivatives[i][j] = nodeValue[i] * layerWeightDerivative[j]

            
    def applyDerivatives(self, learningRate):
        for layer in self.layers:
            layer.applyDerivatives(learningRate)


    def learnFromDataset(self, datasetInput, datasetLabels, testInput, testLabels, learningRate, epochs, threads=1): #add multithreading
        for i in range(epochs):
            trainingLoss = 0
            for j in range(len(datasetInput)):
                computedNetwork = self.computeNetwork(datasetInput[j])
                trainingLoss += sum(self.lossFunctionForList(computedNetwork, datasetLabels[j])) #adds the error
                self.backpropagation(computedNetwork, datasetLabels[j])
                self.applyDerivatives(learningRate)
                if j % 10 == 0:
                    print("progress for epoch: {}%".format((j/len(datasetInput))*100))

            print("MSE for training epoch #{}: {}".format(str(i), trainingLoss/len(datasetInput))) #divides SSE with n inputs for MSE
            
            testLoss = 0
            for j in range(len(testInput)): #computes error for test set
                computedNetwork = self.computeNetwork(testInput[j])
                testLoss += sum(self.lossFunctionForList(computedNetwork, testLabels[j])) #adds the error
            print("MSE for testing epoch #{}: {}".format(str(i), trainingLoss/len(testInput)))


        

    class Layer():
        def __init__(self, inputNodes, outputNodes):
                self.inputLen = inputNodes
                self.outputLen = outputNodes
                self.weights = np.random.random_sample(size = (self.outputLen, self.inputLen)).tolist()
                self.weightDerivatives = np.zeros((self.outputLen, self.inputLen)).tolist()
                self.bias = np.random.random_sample(size=self.outputLen).tolist()
                self.biasDerivatives = np.zeros(self.outputLen).tolist()
                self.previousInputValues = [] * self.inputLen
                self.previousOutputValues = [] * self.outputLen

        def computeLayerValues(self, input): #returns w*a + b for the layer. Corresponds to the z value
            output = []
            self.previousInputValues = [] * self.inputLen #bad code. reused.
            self.previousOutputValues = [] * self.outputLen #bad code. reused.
            self.previousInputValues = input
            for i in range(self.outputLen):
                nodeVal = self.bias[i]
                for j in range(self.inputLen):
                    nodeVal += self.weights[i][j] * input[j]
                output.append(nodeVal)
                self.previousOutputValues.append(nodeVal)
            return output

        def applyDerivatives(self, learningRate):
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    self.weights[i][j] -= self.weightDerivatives[i][j] * learningRate
                self.bias[i] -= self.biasDerivatives[i] * learningRate



def translateLabelsIntoNetwork(labelsList, dimensions):
    out = []
    for label in labelsList:
        lst = [0] * dimensions
        lst[label] = 1
        out.append(lst)
    return out

def scaleTrainingImages(images, scalar):
    out = []
    for image in images:
        out.append(np.multiply(image, scalar))
    return out



mnistData = MNIST()

trainingImages, trainingLabels = mnistData.load_training()

trainingImages = scaleTrainingImages(trainingImages, 1/255)

testingImages, testingLabels = mnistData.load_testing()

testingImages = scaleTrainingImages(testingImages, 1/255)

trainingLabels = translateLabelsIntoNetwork(trainingLabels, 10)

testingLabels = translateLabelsIntoNetwork(testingLabels, 10)


nn = NeuralNet()

nn.addLayer(784,100)

nn.addLayer(100,10)

nn.learnFromDataset(trainingImages[0:2000], trainingLabels[0:2000], testingImages[0:200], testingLabels[0:200], 0.1, 5)



print("done")