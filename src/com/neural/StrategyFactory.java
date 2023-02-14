package com.neural;

public class StrategyFactory
{
    private StrategyFactory(){}

    // expand to a switch case as more strategies are added
    public static Strategy getStrategy(Strategies strategies)
    {
        return new LogisticRegression();
    }
}
