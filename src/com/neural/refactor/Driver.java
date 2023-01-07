package com.neural.refactor;

public class Driver {

    public static void main(String[] args) {

        NeuralNetwork neuralNetwork = new NeuralNetwork(2,2,2);
        neuralNetwork.setInputs(new double[]{0.05,0.10});
        neuralNetwork.setWeights(new double[][]{{0.15,0.20,0.25,0.30},{0.40,0.45,0.50,0.55}});
        neuralNetwork.setBiases(new double[][]{{0.175,0.175},{0.30,0.30}});
        neuralNetwork.train(1000, 0.5f);
    }
}
