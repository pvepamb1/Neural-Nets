package com.neural;

public class StrategyFactory
{
    private StrategyFactory(){}

    // expand to a switch case as more strategies are added
    public static Strategy getStrategy(Activation activation)
    {
        return new LogisticRegression();
    }
}
