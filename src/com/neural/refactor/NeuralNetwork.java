package com.neural.refactor;

/**
 * Based on @see <a href="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/">Backpropagation</a>
 */
public class NeuralNetwork
{
    private final Layers layers;
    private final Strategy strategy;
    private static final Strategies defaultStrategy = Strategies.LOGISTIC_REGRESSION;

    public NeuralNetwork(int... layers)
    {
        this(StrategyFactory.getStrategy(defaultStrategy), layers);
    }

    public NeuralNetwork(Strategy strategy, int... layers)
    {
        this.strategy = strategy;
        this.layers = new Layers(layers);
        strategy.setLayers(this.layers);
    }

    public void train(int epochs, double learningRate)
    {
        for (int i=0; i<epochs; i++)
        {
            forwardPass();
            calculateError();
            backwardPass(learningRate);
        }
    }

    private void forwardPass()
    {
        strategy.forwardPass();
    }

    private void calculateError()
    {
        strategy.calculateError();
    }

    private void backwardPass(double learningRate)
    {
        strategy.backwardPass(learningRate);
    }

    public void setInputs(double[] inputs)
    {
        layers.setInputs(inputs);
    }

    public void setWeights(double[][] weights)
    {
        layers.setWeights(weights);
    }

    public void setBiases(double[][] biases)
    {
        layers.setBiases(biases);
    }

    public void setTargetOutputs(double[] outputs)
    {
        layers.setTargetOutputs(outputs);
    }
}
