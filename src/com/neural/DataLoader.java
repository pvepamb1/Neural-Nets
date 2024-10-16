package com.neural;

public interface DataLoader
{
    boolean hasNext();

    int getLabel(int dataSampleIndex);

    int getDataSampleIndex();

    void resetDataSampleIndex();

    double[][] getNextDataSample();

    int getDatasetSize();

    boolean hasMoreBatches();

    void loadBatch();
}
