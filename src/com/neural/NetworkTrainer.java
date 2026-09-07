package com.neural;

public interface NetworkTrainer
{
    void train(DataLoader dataLoader, Model model, int epochs, int batchSize, double learningRate);
}
