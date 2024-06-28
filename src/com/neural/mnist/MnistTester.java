package com.neural.mnist;

import com.neural.TestStrategy;

public class MnistTester implements TestStrategy
{
    int correct = 0;

    @Override
    public void apply(double[] outputLayer, double[] targetOutput, int label)
    {
        if(label == findMax(outputLayer))
        {
            correct++;
        }
    }

    @Override
    public void printResult()
    {
        int totalDataSamples = MnistDataLoader.getInstance().getDatasetSize();
        int correctlyIdentifiedSamples = correct;
        float accuracy = ((float)correctlyIdentifiedSamples/(float)totalDataSamples) * 100;

        System.out.println("Total: " + totalDataSamples);
        System.out.println("Correct: " + correct);
        System.out.println("Accuracy: " + accuracy);
    }

    private int findMax(double[] output)
    {
        double max = -1000;
        int maxIndex = 0;
        for(int i=0; i< output.length; i++)
        {
            if(output[i] > max)
            {
                max = output[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
