package com.neural;

/**
 * Based on @see <a href="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/">Backpropagation</a>
 */
public class NeuralNetwork
{
    private final Layers layers;
    private final Strategy strategy;
    private static final Strategies defaultStrategy = Strategies.LOGISTIC_REGRESSION;

    private double previousError = Integer.MAX_VALUE;
    private double minError = Integer.MAX_VALUE;
    private double errorRiseFromPreviousCount = 0;
    private double errorRiseFromMinCount = 0;

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
            // Todo: set input and output
            forwardPass();
            calculateError();
            backwardPass(learningRate);
        }

        System.out.println("Error rose " + errorRiseFromPreviousCount + " times from prior error values");
        System.out.println("Error rose " + errorRiseFromMinCount + " times from minimum error");
    }

    private void forwardPass()
    {
        strategy.forwardPass();
    }

    private void calculateError()
    {
        double currentError = strategy.calculateError();

        if(currentError > previousError)
        {
            errorRiseFromPreviousCount++;
        }

        if(currentError > minError)
        {
            errorRiseFromMinCount++;
        }
        else
        {
            minError = currentError;
        }

        previousError = currentError;
    }

    private void backwardPass(double learningRate)
    {
        strategy.backwardPass(learningRate);
    }

    public void setInputs(double[] inputs)
    {
        System.arraycopy(inputs, 0, layers.getInputLayer(), 0, inputs.length);
    }

    public void setWeights(double[][] weights)
    {
        System.arraycopy(weights, 0, layers.getWeights(), 0, weights.length);
    }

    public void setBiases(double[][] biases)
    {
        System.arraycopy(biases, 0, layers.getBiases(), 0, biases.length);
    }

    public void setTargetOutputs(double[] outputs)
    {
        System.arraycopy(outputs, 0, layers.getTargetOutputs(), 0, outputs.length);
    }

    public void assembleLayers()
    {
        layers.assembleLayers();
    }
}
