package com.neural;

public interface DataLoader
{
    int getLabel(int dataSampleIndex);

    int getDataSampleIndex();

    double[][] getNextDataSample();

    int getDatasetSize();
}
