package com.neural.mnist;

import com.neural.DataLoader;

import java.io.IOException;

public class MnistDataLoader implements DataLoader
{
    private static MnistDataLoader mnistDataLoaderInstance;
    private static MnistMatrix[] mnistMatrix;
    private static int currentDataSampleIndex;

    private MnistDataLoader(){}

    public static MnistDataLoader getInstance()
    {
        if(mnistDataLoaderInstance == null)
        {
            mnistDataLoaderInstance = new MnistDataLoader();
        }
        return mnistDataLoaderInstance;
    }

    public static void loadMnistData(String imgDirPath, String labelDirPath)
    {
        try
        {
            mnistMatrix = new MnistDataReader().readData(imgDirPath, labelDirPath);
        }
        catch (IOException e)
        {
            throw new RuntimeException("Check file path"); // Handle this better maybe?
        }
    }

    @Override
    public int getDataSampleIndex()
    {
        return currentDataSampleIndex;
    }

    @Override
    public double[][] getNextDataSample() // everything needs to be rewritten
    {
        MnistMatrix matrix = mnistMatrix[currentDataSampleIndex];

        double[] inputs = new double[matrix.getNumberOfRows() * matrix.getNumberOfColumns()];
        int pos = 0;
        for (int i = 0; i < matrix.getNumberOfRows(); i++)
        {
            for (int j = 0; j < matrix.getNumberOfColumns(); j++)
            {
                inputs[pos] = matrix.getValue(i,j)/255.0;
                pos++;
            }
        }

        double[] outputs = new double[10];
        outputs[matrix.getLabel()] = 1;

        if(currentDataSampleIndex == getDataSampleSize() - 1)
        {
            currentDataSampleIndex = 0;
        }
        else
        {
            currentDataSampleIndex++; // Remember to rewrite when multithreading!
        }

        return new double[][]{inputs, outputs};
    }

    @Override
    public int getDataSampleSize()
    {
        return mnistMatrix.length;
    }
}
