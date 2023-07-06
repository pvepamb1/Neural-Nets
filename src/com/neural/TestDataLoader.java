package com.neural;

public class TestDataLoader implements DataLoader
{
    @Override
    public int getDataSampleIndex()
    {
        return 0;
    }

    @Override
    public double[][] getNextDataSample()
    {
        double[][] inputsAndOutputs = new double[2][];

        inputsAndOutputs[0] = new double[]{0.05, 0.10};
        inputsAndOutputs[1] = new double[]{0.01, 0.99};

        return inputsAndOutputs;
    }

    @Override
    public int getDataSampleSize()
    {
        return 1;
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
