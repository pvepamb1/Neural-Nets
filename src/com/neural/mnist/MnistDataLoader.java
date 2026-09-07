package com.neural.mnist;

import com.neural.DataLoader;

import java.io.IOException;
import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.atomic.AtomicInteger;

// TODO: Rewrite this class to wrap data into an data object instead of delegating it to this class
public class MnistDataLoader implements DataLoader
{
    private static MnistDataLoader mnistDataLoaderInstance;
    private static MnistMatrix[] mnistMatrix;
    private static final AtomicInteger currentDataSampleIndex = new AtomicInteger(0);
    private static final int BATCH_SIZE = 32; // Example batch size // Make it settable!
    private final Queue<double[][]> buffer = new LinkedList<>();

    private MnistDataLoader(){}

    public static synchronized MnistDataLoader getInstance() // Does not need to be synchronized currently
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
    public boolean hasNext()
    {
        return !buffer.isEmpty();
    }

    @Override
    public int getLabel(int dataSampleIndex)
    {
        return mnistMatrix[dataSampleIndex].getLabel();
    }

    @Override
    public int getDataSampleIndex()
    {
        return currentDataSampleIndex.get(); // when using buffer, points to index of last element in buffer + 1
    }

    @Override
    public void resetDataSampleIndex()
    {
        currentDataSampleIndex.set(0);
    }

    @Override
    public synchronized double[][] getNextDataSample()
    {
        return buffer.poll();
    }

    @Override
    public int getDatasetSize()
    {
        return mnistMatrix.length;
    }

    @Override
    public boolean hasMoreBatches()
    {
        return currentDataSampleIndex.get() < getDatasetSize();
    }

    @Override
    public void loadBatch()
    {
        buffer.clear();
        for (int i = 0; i < BATCH_SIZE && currentDataSampleIndex.get() < getDatasetSize(); i++) {
            buffer.add(createDataSample(currentDataSampleIndex.getAndIncrement()));
        }
    }

    private double[][] createDataSample(int index) {
        MnistMatrix matrix = mnistMatrix[index];

        double[] inputs = new double[matrix.getNumberOfRows() * matrix.getNumberOfColumns()];
        int pos = 0;
        for (int i = 0; i < matrix.getNumberOfRows(); i++) {
            for (int j = 0; j < matrix.getNumberOfColumns(); j++) {
                inputs[pos] = matrix.getValue(i, j) / 255.0;
                pos++;
            }
        }

        double[] outputs = new double[10];
        outputs[matrix.getLabel()] = 1;

        double[] label = new double[]{matrix.getLabel()}; // just make this an object!

        return new double[][]{inputs, outputs, label};
    }
}
