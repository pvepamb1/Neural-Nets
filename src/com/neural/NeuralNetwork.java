package com.neural;

import java.util.Arrays;

/**
 * Based on @see <a href="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/">Backpropagation</a>
 */
public class NeuralNetwork
{
    private int[] layers;
    private double[] inputLayer;
    private double[] outputLayer;
    private double[] targetOutputs;
    private double[][] hiddenLayers;
    private double[][][] weights;
    private double[][][] weightGradients;
    private double[][] biases;
    private double[][] biasGradients;
    private double[][] netNeuronToErrorValues;
    private double learningRate;
    private double batchSize;

    private static final Activation defaultActivation = Activation.SIGMOID;

    private double previousError = Integer.MAX_VALUE;
    private double minError = Integer.MAX_VALUE;
    private double errorRiseFromPreviousCount = 0;
    private double errorRiseFromMinCount = 0;

    public NeuralNetwork(int... layers)
    {
        initializeModel(new Layers(layers));
    }

    private void initializeModel(Layers layersObj)
    {
        layers = layersObj.getLayers();
        inputLayer = layersObj.getInputLayer();
        outputLayer = layersObj.getOutputLayer();
        targetOutputs = layersObj.getTargetOutputs();
        hiddenLayers = layersObj.getHiddenLayers();
        weights = layersObj.getWeights();
        weightGradients = layersObj.getWeightGradients();
        biases = layersObj.getBiases();
        biasGradients = layersObj.getBiasGradients();
        netNeuronToErrorValues = layersObj.getNetNeuronToErrorValues();
    }

    public void train(int epochs, int batchSize, double learningRate)
    {
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        int dataSampleLength = 2; // Todo: should be set by a separate class
        for (int i = 0; i < epochs; i++)
        {
            for (int dataSampleIndex = 0; dataSampleIndex < batchSize; dataSampleIndex++)
            {
                // Todo: Input and Output to be set by a loader class
                forwardPass();
                calculateError();
                backwardPass();

                if((dataSampleIndex + 1) % batchSize == 0 || dataSampleIndex + 1 == dataSampleLength)
                {
                    updateWeightsAndBiases(dataSampleIndex);
                }
            }
        }

        System.out.println("Error rose " + errorRiseFromPreviousCount + " times from prior error values");
        System.out.println("Error rose " + errorRiseFromMinCount + " times from minimum error");
    }

    private void calculateError()
    {
        double currentError = calculateCost();

        if(currentError > previousError)
        {
            errorRiseFromPreviousCount++;
        }

        if(currentError > minError)
        {
            errorRiseFromMinCount++;
        }
        else
        {
            minError = currentError;
        }

        previousError = currentError;
    }

    // Todo: The following 4 methods needs to validate the input params. Copy logic from Layers class?
    public void setInputs(double[] inputs)
    {
        inputLayer = inputs;
    }

    public void setTargetOutputs(double[] outputs)
    {
        targetOutputs = outputs;
    }

    // Todo: wrap the following 2 methods into a tester class to limit access
    public void setWeights(double[][][] weights)
    {
        this.weights = weights;
    }

    public void setBiases(double[][] biases)
    {
        this.biases = biases;
    }

    private void forwardPass()
    {
        for (int i = 0; i < layers.length - 1; i++) // for every layer that acts as input
        {
            double[] previousInputLayer = i==0 ? inputLayer : hiddenLayers[i-1]; // hidden layers act as input after 1st iteration
            double[][] weightLayer = weights[i];
            double[] biasLayer = biases[i];
            double[] targetLayer = i == layers.length - 2 ? outputLayer: hiddenLayers[i]; // switch target to output layer for last iteration
            for (int j = 0; j < targetLayer.length; j++) // for every target neuron
            {
                double totalNetInput = 0;
                for (int k = 0; k < previousInputLayer.length; k++) // for every connection
                {
                    totalNetInput += previousInputLayer[k] * weightLayer[j][k];
                }
                totalNetInput += biasLayer[j];
                totalNetInput = 1 / (1 + Math.exp(-totalNetInput));
                targetLayer[j] = totalNetInput;
            }
        }
    }

    private double calculateCost()
    {
        double totalError = 0;
        for (int i = 0; i < outputLayer.length; i++)
        {
            totalError += 0.5 * (Math.pow(targetOutputs[i] - outputLayer[i], 2));
        }
        System.out.println("Total error: " + totalError);

        return totalError;
    }

    private void backwardPass()
    {
        calculateOutputLayerContribution();
        calculateWeightGradientsForOutputLayer();
        calculateWeightGradientsForHiddenLayers();
        //clearNeuronContributions();
        calculateBiasGradientsForOutputLayer();
        calculateBiasGradientsForHiddenLayers();
        clearNetNeuronToErrorValues();
    }

    private void calculateOutputLayerContribution()
    {
        for (int i = 0; i < outputLayer.length; i++)
        {
            double neuron = outputLayer[i];
            double outputChangeForNetInputs = neuron * (1 - neuron);
            double totalErrorChangeForOutput = neuron - targetOutputs[i];
            double neuronContribution = totalErrorChangeForOutput * outputChangeForNetInputs;
            netNeuronToErrorValues[netNeuronToErrorValues.length - 1][i] = neuronContribution;
        }
    }

    private void calculateWeightGradientsForOutputLayer()
    {
        double[][] finalWeightLayer = weights[weights.length - 1];
        double[] previousInputLayer = hiddenLayers[hiddenLayers.length - 1];
        for (int row = 0; row < finalWeightLayer.length; row++)
        {
            double outputNeuronContribution = netNeuronToErrorValues[netNeuronToErrorValues.length -1][row];
            for (int col = 0; col < finalWeightLayer[row].length; col++)
            {
                double previousLayerNeuron = previousInputLayer[col];
                double totalContribution = outputNeuronContribution * previousLayerNeuron;
                weightGradients[weightGradients.length - 1][row][col] = totalContribution;
            }
        }
    }

    private void calculateWeightGradientsForHiddenLayers()
    {
        for (int i = weights.length - 2; i>=0; i--) // for every weight layer but last
        {
            for (int j = 0; j < weights[i].length; j++) // for every weight row (also destNeuronRow)
            {
                for (int k = 0; k < weights[i][j].length; k++) // for every weight (also srcNeuronRow)
                {
                    double[] previousLayer = i!=0 ? hiddenLayers[i - 1] : inputLayer; // Todo: extract to method?
                    double srcNeuron = previousLayer[k];
                    double totalContribution = 0;
                    if(netNeuronToErrorValues[i][j] != 0.0) // the odds the contribution was actually 0?
                    {
                        totalContribution = netNeuronToErrorValues[i][j];
                    }
                    else
                    {
                        double[] destLayer = hiddenLayers[i];
                        double destNeuron = destLayer[j];
                        double outputChangeForNetInputs = destNeuron * (1 - destNeuron);

                        double[] nextLayer = i != hiddenLayers.length - 1 ? hiddenLayers[i + 1] : outputLayer;
                        for (int l = 0; l < nextLayer.length; l++) // 4 nested loops - can we do better?
                        {
                            double nextLayerNeuronContribution = netNeuronToErrorValues[i + 1][l];
                            nextLayerNeuronContribution *= weights[i + 1][l][j]; // too tired to understand index relations
                            totalContribution += nextLayerNeuronContribution;
                        }
                        totalContribution *= outputChangeForNetInputs;
                        netNeuronToErrorValues[i][j] = totalContribution;
                    }
                    weightGradients[i][j][k] = totalContribution * srcNeuron;
                }
            }
        }
    }

    private void calculateBiasGradientsForHiddenLayers()
    {
        for (int i=biases.length-2; i>=0; i--) // for every bias layer but last
        {
            for (int j = 0; j < biases[i].length; j++) // for every bias
            {
                biasGradients[i][j] += netNeuronToErrorValues[i][j];
            }
        }
    }

    private void calculateBiasGradientsForOutputLayer()
    {
        double[] finalBiasLayer = biases[biases.length - 1];
        System.arraycopy(netNeuronToErrorValues[netNeuronToErrorValues.length - 1], 0, biasGradients[biasGradients.length - 1], 0, finalBiasLayer.length);
    }

    private void updateWeightsAndBiases(int dataSampleIndex)
    {
        double divisor = dataSampleIndex + 1 % batchSize == 0 ? batchSize : dataSampleIndex + 1 % batchSize;
        for (int i = 0; i < weightGradients.length; i++)
        {
            for (int j = 0; j < weightGradients[i].length; j++)
            {
                for (int k = 0; k < weightGradients[i][j].length; k++)
                {
                    weights[i][j][k] = weights[i][j][k] - learningRate * (weightGradients[i][j][k] / divisor);
                }
            }
        }

        for (int i = 0; i < biasGradients.length; i++)
        {
            for (int j = 0; j < biasGradients[i].length; j++)
            {
                biases[i][j] = biases[i][j] - learningRate * (biasGradients[i][j] / divisor);
            }
        }
    }

    private void clearNetNeuronToErrorValues()
    {
        for (double[] netNeuronToErrorValue : netNeuronToErrorValues)
        {
            Arrays.fill(netNeuronToErrorValue, 0);
        }
    }
}
