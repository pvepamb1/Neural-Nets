package com.neural;

public interface DataLoader
{
    int getDataSampleIndex();
    double[][] getNextDataSample();
    int getDataSampleSize();
}
