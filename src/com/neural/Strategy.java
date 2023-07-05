package com.neural;

public interface Strategy
{
    void setLayers(Layers layersObj);
    double[] forwardPass();
    double calculateCost();
    void backwardPass(double learningRate, int length, int j);
}
