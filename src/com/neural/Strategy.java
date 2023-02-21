package com.neural;

public interface Strategy
{
    void setLayers(Layers layersObj);
    void forwardPass();
    double calculateCost();
    void backwardPass(double learningRate);
}
