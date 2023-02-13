package com.neural;

import java.util.Random;

public class Layers
{
    private int[] layers;
    private double[] inputLayer;
    private double[] outputLayer;
    private double[] targetOutputs;
    private double[][] hiddenLayers;
    private double[][] weights;
    private double[][] newWeights;
    private double[][] biases;
    private double[][] neuralNetwork;

    public Layers(int... layers)
    {
        validateLayers(layers);
        initializeLayers(layers);
        assembleLayers();
    }

    private void validateLayers(int... layers)
    {
        if (layers.length < 3)
        {
            throw new IllegalArgumentException("Please specify at least 3 layers");
        }

        for (int layer : layers)
        {
            if (layer <= 0)
            {
                throw new IllegalArgumentException("Please specify only non-zero positive values");
            }
        }
    }

    private void initializeLayers(int... layers)
    {
        this.layers = layers;
        int noOfInputs = layers[0];
        int noOfHiddenLayers = layers.length - 2;
        int noOfWeightLayers = noOfHiddenLayers + 1;
        int noOfBiasLayers = noOfHiddenLayers + 1;
        int noOfOutputs = layers[layers.length - 1];
        int totalLayers = noOfHiddenLayers + noOfWeightLayers + noOfBiasLayers + 2;

        neuralNetwork = new double[totalLayers][];

        initializeInputLayer(noOfInputs);
        initializeHiddenLayers(noOfHiddenLayers);
        initializeWeights(noOfWeightLayers);
        initializeNewWeights(noOfWeightLayers);
        initializeBiases(noOfBiasLayers);
        initializeOutputLayer(noOfOutputs);
        initializeTargetOutputLayer(noOfOutputs);
    }

    private void initializeInputLayer(int noOfInputs)
    {
        inputLayer = new double[noOfInputs];
    }

    private void initializeHiddenLayers(int noOfHiddenLayers)
    {
        hiddenLayers = new double[noOfHiddenLayers][];
        for (int i = 0; i < noOfHiddenLayers; i++)
        {
            hiddenLayers[i] = new double[layers[i + 1]];
        }
    }

    private void initializeWeights(int noOfWeightLayers)
    {
        weights = new double[noOfWeightLayers][];
        for (int i = 0; i < noOfWeightLayers; i++)
        {
            weights[i] = getRandomDoubles((long) layers[i] * layers[i + 1]);
        }
    }

    private void initializeNewWeights(int noOfWeightLayers)
    {
        newWeights = new double[noOfWeightLayers][];
        for (int i = 0; i < noOfWeightLayers; i++)
        {
            newWeights[i] = new double[layers[i] * layers[i + 1]];
        }
    }

    private void initializeBiases(int noOfBiasLayers)
    {
        biases = new double[noOfBiasLayers][];
        for (int i = 0; i < noOfBiasLayers; i++)
        {
            biases[i] = getRandomDoubles(layers[i + 1]);
        }
    }

    private double[] getRandomDoubles(long streamSize)
    {
        return new Random().doubles(streamSize, 0, 1).toArray();
    }

    private void initializeOutputLayer(int noOfOutputs)
    {
        outputLayer = new double[noOfOutputs];
    }

    private void initializeTargetOutputLayer(int noOfOutputs)
    {
        targetOutputs = new double[noOfOutputs];
    }

    public void assembleLayers()
    {
        neuralNetwork[0] = inputLayer;

        for (int i = 0; i < weights.length; i++)
        {
            neuralNetwork[i * 3 + 1] = weights[i];
        }

        for (int i = 0; i < biases.length; i++)
        {
            neuralNetwork[i * 3 + 2] = biases[i];
        }

        for (int i = 0; i < hiddenLayers.length; i++)
        {
            neuralNetwork[i * 3 + 3] = hiddenLayers[i];
        }

        neuralNetwork[neuralNetwork.length - 1] = outputLayer;
    }

    public double[] getInputLayer()
    {
        return inputLayer;
    }

    public void setInputs(double[] inputs)
    {
        if (inputs.length != inputLayer.length)
        {
            throw new IllegalArgumentException("Mismatch in specified input size " +
                    "and provided input. Difference: " + (inputs.length - inputLayer.length));
        }
        for (int i = 0; i < inputs.length; i++)
        {
            neuralNetwork[0] = inputs;
        }
    }

    public double[][] getWeights()
    {
        return weights;
    }

    public void setWeights(double[][] weights)
    {
        if (weights.length != this.weights.length)
        {
            throw new IllegalArgumentException("Incorrect no. of weights provided. Expected: "
                    + this.weights.length + ", provided: " + weights.length);
        }

        for (int i = 0; i < weights.length; i++)
        {
            if (this.weights[i].length != weights[i].length)
            {
                throw new IllegalArgumentException("Incorrect no. of weights provided for layer: "
                        + i + ". Expected: " + this.weights.length + ", provided: " + weights.length);
            }
            neuralNetwork[i * 3 + 1] = weights[i];
        }
    }

    public double[][] getBiases()
    {
        return biases;
    }

    public void setBiases(double[][] biases)
    {
        if (biases.length != this.biases.length)
        {
            throw new IllegalArgumentException("Incorrect no. of biases provided. Expected: "
                    + this.biases.length + ", provided: " + biases.length);
        }

        for (int i = 0; i < biases.length; i++)
        {
            if (this.biases[i].length != biases[i].length)
            {
                throw new IllegalArgumentException("Incorrect no. of biases provided for layer: "
                        + i + ". Expected: " + this.biases.length + ", provided: " + biases.length);
            }
            neuralNetwork[i * 3 + 2] = biases[i];
        }
    }

    public double[] getTargetOutputs()
    {
        return targetOutputs;
    }

    public void setTargetOutputs(double[] outputs)
    {
        if (outputs.length != outputLayer.length)
        {
            throw new IllegalArgumentException("Mismatch in specified output size " +
                    "and provided target outputs. Difference: " + (outputs.length - outputLayer.length));
        }
        for (int i = 0; i < outputs.length; i++)
        {
            targetOutputs = outputs;
        }
    }

    // all getters in this class exist to make extracting layers info easier. Maybe a utility method to replace these?
    public int[] getLayers()
    {
        return layers;
    }

    public double[] getOutputLayer()
    {
        return outputLayer;
    }

    public double[][] getHiddenLayers()
    {
        return hiddenLayers;
    }

    public double[][] getNewWeights()
    {
        return newWeights;
    }

    public double[][] getNeuralNetwork()
    {
        return neuralNetwork;
    }
}
