package com.neural;

public class SimpleTrainer implements NetworkTrainer
{
    @Override
    public void train(DataLoader dataLoader, Model model, int epochs, int batchSize, double learningRate)
    {
        NeuralNetwork neuralNetwork = new NeuralNetwork(dataLoader, model);
        neuralNetwork.train(epochs, batchSize, learningRate);
    }
}
