package com.neural;

import com.neural.mnist.MnistNeuralNetwork;

public class Driver
{
    public static void main(String[] args)
    {
        long start = System.currentTimeMillis();

        //NeuralNetwork neuralNetwork = new NeuralNetwork(InputType.TEST, 2, 2, 2);
        //neuralNetwork.train(10000, 1, 0.5f);

        MnistNeuralNetwork mnistNeuralNetwork = new MnistNeuralNetwork("/Users/prasenna/Downloads/train-images.idx3-ubyte","/Users/prasenna/Downloads/train-labels.idx1-ubyte",  32);
        mnistNeuralNetwork.train(10, 32, 1);
        mnistNeuralNetwork.test("/Users/prasenna/Downloads/t10k-images.idx3-ubyte","/Users/prasenna/Downloads/t10k-labels.idx1-ubyte");

        long stop = System.currentTimeMillis();
        printTimeTaken(start, stop);
    }

    private static void printTimeTaken(long start, long stop)
    {
        long millis = stop - start;
        long minutes = (millis / 1000)  / 60;
        int seconds = (int)((millis / 1000) % 60);
        System.out.println("Completed in " + minutes + " minutes " + seconds + " seconds and " + millis % 1000 + " milliseconds");
    }
}
