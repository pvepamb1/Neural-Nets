package com.neural;

public class CustomDataLoader implements DataLoader
{
    public static boolean forTraining = true;
    private int nextDataSampleIndex;

    @Override
    public boolean hasNext()
    {
        return getDatasetSize() >= nextDataSampleIndex;
    }

    @Override
    public int getLabel(int dataSampleIndex)
    {
        return 0;
    }

    @Override
    public int getDataSampleIndex()
    {
        return nextDataSampleIndex;
    }

    @Override
    public void resetDataSampleIndex()
    {

    }

    @Override
    public double[][] getNextDataSample()
    {
        double[][] inputsAndOutputs = new double[2][];

        double[][] inputs = getInputs();
        double[][] outputs = getOutputs();

        inputsAndOutputs[0] = inputs[nextDataSampleIndex];
        inputsAndOutputs[1] = outputs[nextDataSampleIndex];

        if(nextDataSampleIndex == getDatasetSize() - 1)
        {
            nextDataSampleIndex = 0;
        }
        else
        {
            nextDataSampleIndex++; // Remember to rewrite when multithreading!
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
    public int getDatasetSize()
    {
        return getInputs().length;
    }

    @Override
    public boolean hasMoreBatches()
    {
        return false;
    }

    @Override
    public void loadBatch()
    {

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
