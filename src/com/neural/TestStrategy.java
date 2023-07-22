package com.neural;

public interface TestStrategy
{
    void apply(double[] outputLayer, double[] targetOutput, int label);

    void printResult();
}
