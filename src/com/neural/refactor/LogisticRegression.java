package com.neural.refactor;

public class LogisticRegression implements Strategy
{
    private int[] layers;
    private double[] inputLayer;
    private double[] outputLayer;
    private double[] targetOutputs;
    private double[][] hiddenLayers;
    private double[][] weights;
    private double[][] newWeights;
    private double[][] biases;
    // Todo: Nice to have but necessary? Might be easier to use layers directly for calculations
    private double[][] neuralNetwork;

    @Override
    public void setLayers(Layers layersObj)
    {
        layers = layersObj.getLayers();
        inputLayer = layersObj.getInputLayer();
        outputLayer = layersObj.getOutputLayer();
        targetOutputs = layersObj.getTargetOutputs();
        hiddenLayers = layersObj.getHiddenLayers();
        weights = layersObj.getWeights();
        newWeights = layersObj.getNewWeights();
        biases = layersObj.getBiases();
        neuralNetwork = layersObj.getNeuralNetwork();
    }

    @Override
    public void forwardPass()
    {
        calculateOutput();
    }

    private void calculateOutput()
    {
        for (int i = 3; i < neuralNetwork.length; i += 3)
        {
            double[] previousInputLayer = neuralNetwork[i - 3];
            double[] previousWeightLayer = neuralNetwork[i - 2];
            double[] previousBiasLayer = neuralNetwork[i - 1];
            for (int j = 0; j < neuralNetwork[i].length; j++)
            {
                double totalNetInput = 0;
                for (int k = 0; k < previousInputLayer.length; k++)
                {
                    int weightArrayOffset = j * previousInputLayer.length;
                    totalNetInput += previousInputLayer[k] * previousWeightLayer[k + weightArrayOffset]
                            + previousBiasLayer[j];
                }
                totalNetInput = 1 / (1 + Math.exp(-totalNetInput));
                neuralNetwork[i][j] = totalNetInput;
            }
        }
    }

    // trying to avoid using the neural network object for computation
    private void calculateOutputForHiddenLayer()
    {
        for (int i=0; i<hiddenLayers.length; i++)
        {
            for (int j = 0; j < hiddenLayers[i].length; j++)
            {
                double[] previousInputLayer = i!=0 ? hiddenLayers[i-1] : inputLayer;
                double totalNetInput = 0;
                for (int k = 0; k < weights[i].length; k++)
                {
                    totalNetInput += previousInputLayer[j] * weights[i][k] + biases[i][j];
                }
                totalNetInput = 1 / (1 + Math.exp(-totalNetInput));
                hiddenLayers[i][j] = totalNetInput;
            }
        }
    }

    private void calculateOutputForOutputLayer()
    {

    }

    @Override
    public void calculateError()
    {
        double totalError = 0;
        for (int i = 0; i < outputLayer.length; i++)
        {
            totalError += 0.5 * (Math.pow(targetOutputs[i] - neuralNetwork[neuralNetwork.length - 1][i], 2));
        }
        //System.out.println("Total error: " + totalError);
    }

    @Override
    public void backwardPass(double learningRate)
    {
        newWeights = new double[weights.length][];
        calculateNewWeightsForOutputLayer(learningRate);
        calculateNewWeightsForHiddenLayers();
    }

    private void calculateNewWeightsForOutputLayer(double learningRate)
    {
        // initialize the last layer with length equalling weights' last layer
        // Todo: Move initialization to its own method further up like other layers?
        newWeights[newWeights.length -1] = new double[weights[weights.length - 1].length];
        for(int i=0; i<outputLayer.length; i++)
        {
            double output = neuralNetwork[neuralNetwork.length - 1][i];
            double totalErrorChangeForOutput = output - targetOutputs[i];
            double outputChangeForNetInputs = output*(1 - output);
            double[] previousWeightLayer = neuralNetwork[neuralNetwork.length - 3];
            double[] previousInputLayer = neuralNetwork[neuralNetwork.length - 4];
            int weightArrayOffset = previousInputLayer.length;
            for (int j = i * weightArrayOffset; j < previousInputLayer.length + i * weightArrayOffset; j++)
            {
                double netInputChangeForWeight = previousInputLayer[i]; // this is wrong. value should be the previous input value connected to (j)
                double totalErrorChangeForWeight = totalErrorChangeForOutput * outputChangeForNetInputs
                        * netInputChangeForWeight;
                newWeights[newWeights.length - 1][j] = previousWeightLayer[j] - learningRate * totalErrorChangeForWeight;
            }
        }
    }

    private void calculateNewWeightsForHiddenLayers()
    {
        for (int i=neuralNetwork.length-4; i>=0; i-=3)
        {
            for (int j = 0; j < neuralNetwork[i].length; j++)
            {

            }
        }
    }
}
