package com.neural.refactor;

public interface Strategy
{
    void setLayers(Layers layersObj);
    void forwardPass();
    double calculateError();
    void backwardPass(double learningRate);
}
