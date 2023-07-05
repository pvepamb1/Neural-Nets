package com.neural;

import com.neural.mnist.MnistDataReader;
import com.neural.mnist.MnistMatrix;

import java.io.IOException;

public class Driver
{
    public static void main(String[] args) throws IOException
    {
        long start = System.currentTimeMillis();
        //NeuralNetwork neuralNetwork = new NeuralNetwork(2,2,2);
        NeuralNetwork neuralNetwork = new NeuralNetwork("/Users/prasenna/Downloads/train-images.idx3-ubyte", "/Users/prasenna/Downloads/train-labels.idx1-ubyte", true, 16, 16);
        //setTestData(neuralNetwork);
        neuralNetwork.train(50, 0.05);
        System.out.println(" --------- ");
        MnistDataReader dataReader = new MnistDataReader();
        MnistMatrix[] matrices = dataReader.readData("/Users/prasenna/Downloads/t10k-images.idx3-ubyte", "/Users/prasenna/Downloads/t10k-labels.idx1-ubyte");
        //MnistMatrix matrix = matrices[1];
        int correct = 0;
        for (MnistMatrix matrix: matrices)
        {
            if(neuralNetwork.test(matrix))
                correct++;
        }
        System.out.println("Total: " + matrices.length);
        System.out.println("Correct: " + correct);
        System.out.println("Accuracy: " + (100 * correct)/ matrices.length);
        long stop = System.currentTimeMillis();
        long millis = stop - start;
        long minutes = (millis / 1000)  / 60;
        int seconds = (int)((millis / 1000) % 60);
        System.out.println("Took: " + minutes + " minutes and " + seconds + " seconds");
    }

    public static void setTestData(NeuralNetwork neuralNetwork)
    {
        neuralNetwork.setInputs(new double[]{0.05, 0.10});
        neuralNetwork.setWeights(new double[][][]{{{0.15, 0.20}, {0.25, 0.30}}, {{0.40, 0.45}, {0.50, 0.55}}});
        neuralNetwork.setBiases(new double[][]{{0.35, 0.35}, {0.60, 0.60}});
        neuralNetwork.setTargetOutputs(new double[]{0.01, 0.99});
        neuralNetwork.assembleLayers();
    }
}
