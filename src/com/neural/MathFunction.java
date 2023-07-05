package com.neural;

import java.util.function.Function;

public interface MathFunction
{
    Function<Double, Double> getEquation();
    Function<Double, Double> getDerivativeWRTWeight();
}
