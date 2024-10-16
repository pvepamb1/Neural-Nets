package com.neural.concurrent;

import com.neural.Model;

public class ForwardPropagation
{
    public static void forwardPass(Model model)
    {
        int[] layers = model.getLayers();
        double[] inputLayer = model.getInputLayer();
        double[] outputLayer = model.getOutputLayer();
        double[][] hiddenLayers = model.getHiddenLayers();
        double[][][] weights = model.getWeights();
        double[][] biases = model.getBiases();

        for (int i = 0; i < layers.length - 1; i++) // for every layer that acts as input
        {
            double[] previousInputLayer = i == 0 ? inputLayer : hiddenLayers[i - 1]; // hidden layers act as input after 1st iteration
            double[][] weightLayer = weights[i];
            double[] biasLayer = biases[i];
            double[] targetLayer = i == layers.length - 2 ? outputLayer : hiddenLayers[i]; // switch target to output layer for last iteration
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
}
