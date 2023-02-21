package com.neural;

import java.util.Arrays;

public class Sigmoid implements Strategy
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
    private double[][] neuronContributions;
    private double learningRate;
    // Todo: needs to be initialized via constructor
    private double batchSize = 1;

    @Override
    public void setLayers(Layers layersObj)
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
        neuronContributions = layersObj.getNeuronContributions();
    }

    @Override
    public void forwardPass()
    {
        calculateOutput();
    }

    private void calculateOutput()
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

    @Override
    public double calculateCost()
    {
        double totalError = 0;
        for (int i = 0; i < outputLayer.length; i++)
        {
            totalError += 0.5 * (Math.pow(targetOutputs[i] - outputLayer[i], 2));
        }
        System.out.println("Total error: " + totalError);

        return totalError;
    }

    @Override
    public void backwardPass(double learningRate)
    {
        this.learningRate = learningRate; // Todo: should be moved up to constructor
        calculateOutputLayerContribution();
        calculateWeightGradientsForOutputLayer();
        calculateWeightGradientsForHiddenLayers();
        //clearNeuronContributions();
        calculateNewBiasesForOutputLayer();
        calculateNewBiasesForHiddenLayers();
        clearNeuronContributions();
        updateWeightsAndBiases();
    }

    private void calculateOutputLayerContribution()
    {
        for (int i = 0; i < outputLayer.length; i++)
        {
            double neuron = outputLayer[i];
            double outputChangeForNetInputs = neuron * (1 - neuron);
            double totalErrorChangeForOutput = neuron - targetOutputs[i];
            double neuronContribution =  totalErrorChangeForOutput * outputChangeForNetInputs;
            neuronContributions[neuronContributions.length - 1][i] = neuronContribution;
        }
    }

    private void calculateWeightGradientsForOutputLayer()
    {
        double[][] finalWeightLayer = weights[weights.length - 1];
        double[] previousInputLayer = hiddenLayers[hiddenLayers.length - 1];
        for (int row = 0; row < finalWeightLayer.length; row++)
        {
            double outputNeuronContribution = neuronContributions[neuronContributions.length -1][row];
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
                    if(neuronContributions[i][j] != 0.0) // the odds the contribution was actually 0?
                    {
                        totalContribution = neuronContributions[i][j];
                    }
                    else
                    {
                        double[] destLayer = hiddenLayers[i];
                        double destNeuron = destLayer[j];
                        double outputChangeForNetInputs = destNeuron * (1 - destNeuron);

                        double[] nextLayer = i != hiddenLayers.length - 1 ? hiddenLayers[i + 1] : outputLayer;
                        for (int l = 0; l < nextLayer.length; l++) // 4 nested loops - can we do better?
                        {
                            double nextLayerNeuronContribution = neuronContributions[i + 1][l];
                            nextLayerNeuronContribution *= weights[i + 1][l][j]; // too tired to understand index relations
                            totalContribution += nextLayerNeuronContribution;
                        }
                        totalContribution *= outputChangeForNetInputs;
                        neuronContributions[i][j] = totalContribution;
                    }
                    weightGradients[i][j][k] = totalContribution * srcNeuron;
                }
            }
        }
    }

    private void calculateNewBiasesForHiddenLayers()
    {
        for (int i=biases.length-2; i>=0; i--) // for every bias layer but last
        {
            for (int j = 0; j < biases[i].length; j++) // for every bias
            {
                //double currentBias = neuralNetwork[i][j];
                biasGradients[i][j] += neuronContributions[i][j];
            }
        }
    }

    private void calculateNewBiasesForOutputLayer()
    {
        double[] finalBiasLayer = biases[biases.length - 1];
        System.arraycopy(neuronContributions[neuronContributions.length - 1], 0, biasGradients[biasGradients.length - 1], 0, finalBiasLayer.length);
    }

    private void updateWeightsAndBiases()
    {
        for (int i = 0; i < weightGradients.length; i++)
        {
            for (int j = 0; j < weightGradients[i].length; j++)
            {
                for (int k = 0; k < weightGradients[i][j].length; k++)
                {
                    weights[i][j][k] = weights[i][j][k] - learningRate * (weightGradients[i][j][k] / batchSize);
                }
            }
        }

        for (int i = 0; i < biasGradients.length; i++)
        {
            for (int j = 0; j < biasGradients[i].length; j++)
            {
                biases[i][j] = biases[i][j] - learningRate * (biasGradients[i][j] / batchSize);
            }
        }
    }

    private void clearNeuronContributions()
    {
        for (double[] neuronContribution : neuronContributions)
        {
            Arrays.fill(neuronContribution, 0);
        }
    }
}
