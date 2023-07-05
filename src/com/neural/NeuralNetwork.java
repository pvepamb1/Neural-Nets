package com.neural;

import com.neural.mnist.MnistDataReader;
import com.neural.mnist.MnistMatrix;

import java.io.IOException;

/**
 * Based on @see <a href="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/">Backpropagation</a>
 */
public class NeuralNetwork
{
    private final Layers layers;
    private Strategy strategy;
    private static final Activation DEFAULT_ACTIVATION_FUNCTION = Activation.SIGMOID;

    private double previousError = Integer.MAX_VALUE;
    private double minError = Integer.MAX_VALUE;
    private double errorRiseFromPreviousCount = 0;
    private double errorRiseFromMinCount = 0;

    private static MnistMatrix[] mnistMatrix;

    public NeuralNetwork(int... layers)
    {
        this(StrategyFactory.getStrategy(DEFAULT_ACTIVATION_FUNCTION), layers);
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
            for (int j = 0; j<mnistMatrix.length; j++)
            {
                setInputsAndOutputsForRow(j);
                forwardPass();
                calculateError();
                backwardPass(learningRate, mnistMatrix.length, j);
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
        double currentError = strategy.calculateCost();

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

    private void backwardPass(double learningRate, int length, int j)
    {
        strategy.backwardPass(learningRate, length, j);
    }

    private void setInputsAndOutputsForRow(int mnistRow)
    {
        MnistMatrix matrix = mnistMatrix[mnistRow];
        setInputsAndOutputs(matrix);
    }

    private void setInputsAndOutputs(MnistMatrix matrix)
    {
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

    public void setStrategy(Activation strategy)
    {
        Strategy strategy1 = StrategyFactory.getStrategy(strategy);
        this.strategy = strategy1;
        strategy1.setLayers(layers);
    }

    public boolean test(MnistMatrix matrix)
    {
        setInputsAndOutputs(matrix);
        double[] output = strategy.forwardPass();
        int actual = matrix.getLabel();
        int predicted = findMax(output);

        return actual == predicted;
        /*for(double out: output)
        {
            System.out.print(out + " ");
        }
        System.out.println();*/
    }

    private int findMax(double[] output)
    {
        double max = -1000;
        int maxIndex = 0;
        for(int i=0; i< output.length; i++)
        {
            if(output[i] > max)
            {
                max = output[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void getTargetOutputs()
    {
        for (double output: layers.getTargetOutputs())
        {
            System.out.print(output + " ");
        }
    }

    public void setInputs(double[] inputs)
    {
        System.arraycopy(inputs, 0, layers.getInputLayer(), 0, inputs.length);
    }

    public void setWeights(double[][][] weights)
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
