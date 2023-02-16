package com.neural;

import com.neural.mnist.MnistDataReader;
import com.neural.mnist.MnistMatrix;

import java.io.IOException;

/**
 * Based on @see <a href="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/">Backpropagation</a>
 */
public class NeuralNetwork
{
    private Strategy strategy;
    private final Layers layers;
    private static final Strategies defaultStrategy = Strategies.LOGISTIC_REGRESSION;

    private double previousError = Integer.MAX_VALUE;
    private double minError = Integer.MAX_VALUE;
    private double errorRiseFromPreviousCount = 0;
    private double errorRiseFromMinCount = 0;

    private static MnistMatrix[] mnistMatrix;

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

    public NeuralNetwork(String imgDirPath, String labelDirPath, boolean isMnist, int... hiddenLayers) throws IOException
    {
        this(buildLayerInfo(imgDirPath, labelDirPath, isMnist, hiddenLayers));
    }

    private static int[] buildLayerInfo(String imgDirPath, String labelDirPath, boolean isMnist, int... hiddenLayers) throws IOException
    {
        int[] layers = new int[hiddenLayers.length + 2];
        if(isMnist)
        {
            mnistMatrix = new MnistDataReader().readData(imgDirPath, labelDirPath);
            layers[0] = mnistMatrix[mnistMatrix.length - 1].getNumberOfRows()
                    * mnistMatrix[mnistMatrix.length - 1].getNumberOfColumns();
            layers[layers.length - 1] = 10;
        }
        System.arraycopy(hiddenLayers, 0, layers, 1, hiddenLayers.length);
        return layers;
    }

    public void train(int epochs, double learningRate)
    {
        for (int i=0; i<epochs; i++)
        {
            for (int j = 0; j<1000; j++)
            {
                setInputsAndOutputs(j);
                forwardPass();
                calculateError();
                backwardPass(learningRate);
            }
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

    private void setInputsAndOutputs(int mnistRow)
    {
        MnistMatrix matrix = mnistMatrix[mnistRow];
        double[] inputs = new double[matrix.getNumberOfRows() * matrix.getNumberOfColumns()];
        int pos = 0;
        for (int i = 0; i < matrix.getNumberOfRows(); i++)
        {
            for (int j = 0; j < matrix.getNumberOfColumns(); j++)
            {
                inputs[pos] = matrix.getValue(i,j)/255.0;
                pos++;
            }
        }
        setInputs(inputs);
        double[] outputs = new double[10];
        outputs[matrix.getLabel()] = 1;
        setTargetOutputs(outputs);
    }

    public void setStrategy(Strategies strategy)
    {
        Strategy strategy1 = StrategyFactory.getStrategy(strategy);
        this.strategy = strategy1;
        strategy1.setLayers(layers);
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
