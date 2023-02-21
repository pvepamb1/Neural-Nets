package com.neural;

public class Driver
{
    public static void main(String[] args)
    {
        long start = System.currentTimeMillis();
        NeuralNetwork neuralNetwork = new NeuralNetwork(2, 2, 3, 2);
        setTestData(neuralNetwork);
        neuralNetwork.train(10000, 0.5f);
        long stop = System.currentTimeMillis();
        System.out.println(stop - start);
    }

    public static void setTestData(NeuralNetwork neuralNetwork)
    {
        neuralNetwork.setInputs(new double[]{0.05, 0.10});
        neuralNetwork.setWeights(new double[][][]{{{0.15, 0.20}, {0.25, 0.30}}, {{0.40, 0.45}, {0.50, 0.55}, {0.60, 0.65}}, {{0.75, 0.80, 0.85}, {0.90, 0.95, 1.0}}});
        neuralNetwork.setBiases(new double[][]{{0.35, 0.35}, {0.90, 0.90, 0.90}, {0.20, 0.20}});
        neuralNetwork.setTargetOutputs(new double[]{0.01, 0.99});
        neuralNetwork.assembleLayers();
    }
}
