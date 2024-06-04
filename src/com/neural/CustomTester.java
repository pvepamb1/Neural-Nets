package com.neural;

public class CustomTester implements TestStrategy
{
    @Override
    public void apply(double[] outputLayer, double[] targetOutput, int label)
    {
        if(outputLayer[0] == targetOutput[0])
        {
            System.out.println("success");
        }
    }

    @Override
    public void printResult()
    {

    }
}
