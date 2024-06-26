package com.neural;

public class CustomDataLoader implements DataLoader
{
    public static boolean forTraining = true;
    private int currentDataSampleIndex;

    @Override
    public int getLabel(int dataSampleIndex)
    {
        return 0;
    }

    @Override
    public int getDataSampleIndex()
    {
        return currentDataSampleIndex;
    }

    @Override
    public double[][] getNextDataSample()
    {
        double[][] inputsAndOutputs = new double[2][];

        double[][] inputs = getInputs();
        double[][] outputs = getOutputs();

        inputsAndOutputs[0] = inputs[currentDataSampleIndex];
        inputsAndOutputs[1] = outputs[currentDataSampleIndex];

        if(currentDataSampleIndex == getDataSampleSize() - 1)
        {
            currentDataSampleIndex = 0;
        }
        else
        {
            currentDataSampleIndex++; // Remember to rewrite when multithreading!
        }

        return inputsAndOutputs;
    }

    private double[][] getInputs()
    {
        if(forTraining)
        {
            return new double[][]{{0.05, 0.10}};
        }
        else
        {
            return new double[][]{{5}};
        }
    }

    private double[][] getOutputs()
    {
        if(forTraining)
        {
            return new double[][]{{0.01, 0.99}};
        }
        else
        {
            return new double[][]{{10}};
        }
    }

    @Override
    public int getDataSampleSize()
    {
        return getInputs().length;
    }

    public double[][][] getWeights()
    {
        return new double[][][]{{{0.15, 0.20}, {0.25, 0.30}}, {{0.40, 0.45}, {0.50, 0.55}}};
    }

    public double[][] getBiases()
    {
        return new double[][]{{0.35, 0.35}, {0.60, 0.60}};
    }
}
