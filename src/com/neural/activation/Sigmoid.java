package com.neural.activation;

import com.neural.function.NeuralNetworkFunction;

import java.util.function.Function;

public class Sigmoid implements NeuralNetworkFunction
{
    @Override
    public Function<Double, Double> getEquation()
    {
        return x -> 1 / (1 + Math.exp(-x));
    }

    @Override
    public Function<Double, Double> getErrorPartialDerivativeWRTWeight()
    {
        return null;
    }
}
