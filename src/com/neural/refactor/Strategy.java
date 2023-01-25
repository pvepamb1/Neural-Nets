package com.neural.refactor;

public interface Strategy
{
    void setLayers(Layers layersObj);
    void forwardPass();
    void calculateError();
    void backwardPass(double learningRate);
}
