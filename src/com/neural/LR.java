package com.neural;

import java.util.function.Function;

public class LR implements MathFunction
{
    @Override
    public Function<Double, Double> getEquation()
    {
        return x -> 1 / (1 + Math.exp(-x));
    }

    @Override
    public Function<Double, Double> getDerivativeWRTWeight()
    {
        return null;
    }
}
