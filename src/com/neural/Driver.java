package com.neural;

import com.neural.mnist.MnistNeuralNetwork;

public class Driver
{
    public static void main(String[] args)
    {
        long start = System.currentTimeMillis();

        //NeuralNetwork neuralNetwork = new NeuralNetwork(InputType.CUSTOM, 1,1);
        //neuralNetwork.train(10, 1, 0.11);
        //neuralNetwork.test(new CustomTester());

        MnistNeuralNetwork mnistNeuralNetwork = new MnistNeuralNetwork("mnistData/train-images.idx3-ubyte","mnistData/train-labels.idx1-ubyte",  32);
        mnistNeuralNetwork.train(1, 32, 1);
        mnistNeuralNetwork.test("mnistData/t10k-images.idx3-ubyte","mnistData/t10k-labels.idx1-ubyte");

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
