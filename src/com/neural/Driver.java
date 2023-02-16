package com.neural;

import java.io.IOException;

public class Driver
{
    public static void main(String[] args) throws IOException
    {
        NeuralNetwork neuralNetwork = new NeuralNetwork("/Users/prasenna/Downloads/train-images.idx3-ubyte", "/Users/prasenna/Downloads/train-labels.idx1-ubyte", true, 16, 16);
        //setTestData(neuralNetwork);
        neuralNetwork.train(2, 0.01);
    }

    public static void setTestData(NeuralNetwork neuralNetwork)
    {
        neuralNetwork.setInputs(new double[]{0.05, 0.10});
        neuralNetwork.setWeights(new double[][]{{0.15, 0.20, 0.25, 0.30}, {0.40, 0.45, 0.50, 0.55}});
        neuralNetwork.setBiases(new double[][]{{0.175, 0.175}, {0.30, 0.30}});
        neuralNetwork.setTargetOutputs(new double[]{0.01, 0.99});
        neuralNetwork.assembleLayers();
    }
}
