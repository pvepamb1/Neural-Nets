package com.neural.mnist;

import com.neural.DataLoader;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

// TODO: Rewrite this class to wrap data into an data object instead of delegating it to this class
public class MnistDataLoader implements DataLoader
{
    private static MnistDataLoader mnistDataLoaderInstance;
    private static MnistMatrix[] mnistMatrix;
    private static final AtomicInteger currentDataSampleIndex = new AtomicInteger(0);

    private MnistDataLoader(){}

    public static synchronized MnistDataLoader getInstance()
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
    public int getLabel(int dataSampleIndex)
    {
        return mnistMatrix[dataSampleIndex].getLabel();
    }

    @Override
    public int getDataSampleIndex()
    {
        return currentDataSampleIndex.get();
    }

    @Override
    public synchronized double[][] getNextDataSample() // everything needs to be rewritten
    {
        MnistMatrix matrix = mnistMatrix[currentDataSampleIndex.get()];

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

        if(currentDataSampleIndex.get() == getDatasetSize() - 1)
        {
            currentDataSampleIndex.set(0);
        }
        else
        {
            currentDataSampleIndex.incrementAndGet();
        }

        return new double[][]{inputs, outputs};
    }

    @Override
    public int getDatasetSize()
    {
        return mnistMatrix.length;
    }
}
