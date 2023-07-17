package com.neural.function;

import java.util.function.Function;

public interface NeuralNetworkFunction extends MathFunction
{
    Function<Double, Double> getErrorPartialDerivativeWRTWeight();
}
