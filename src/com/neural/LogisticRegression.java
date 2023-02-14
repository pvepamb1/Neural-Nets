package com.neural;

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
    private double[][] newBiases;
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
        newBiases = layersObj.getNewBiases();
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

    @Override
    public double calculateError()
    {
        double totalError = 0;
        for (int i = 0; i < outputLayer.length; i++)
        {
            totalError += 0.5 * (Math.pow(targetOutputs[i] - neuralNetwork[neuralNetwork.length - 1][i], 2));
        }
        System.out.println("Total error: " + totalError);

        return totalError;
    }

    @Override
    public void backwardPass(double learningRate)
    {
        calculateNewWeightsForOutputLayer(learningRate);
        calculateNewWeightsForHiddenLayers(learningRate);
        calculateNewBiasesForOutputLayer(learningRate);
        calculateNewBiasesForHiddenLayers(learningRate);
        updateWeightsAndBiases();
    }

    private void calculateNewWeightsForHiddenLayers(double learningRate)
    {
        for (int i=neuralNetwork.length-6; i>=0; i-=3) // for every weight layer but last
        {
            for (int j = 0; j < neuralNetwork[i].length; j++) // for every weight
            {
                double currentWeight = neuralNetwork[i][j];
                int[] destNeuronIndex = getDestNeuronIndexForWeight(i, j);
                int destNeuronIndexCol = destNeuronIndex[0];
                int destNeuronIndexRow = destNeuronIndex[1];
                double srcNeuron = getSrcNeuronValueForWeight(i, j);
                newWeights[(i-1)/3][j] = currentWeight
                        - calculateNeuronContribution(destNeuronIndexCol, destNeuronIndexRow, false)
                        * srcNeuron * learningRate;
            }
        }
    }

    private double calculateNeuronContribution(int col, int row, boolean isBias)
    {
        if(col == neuralNetwork.length - 1)
        {
           return calculateOutputLayerContribution(row);
        }

        double[] nextLayer = neuralNetwork[col + 3];
        double currentNeuron = neuralNetwork[col][row];
        double outputChangeForNetInputs = currentNeuron * (1 - currentNeuron);
        double totalContribution = 0;
        for (int i = 0; i < nextLayer.length; i++)
        {
            double netInput = calculateNeuronContribution(col + 3, i, isBias);

            // the partial derivative of bias wrt net input is 1 and so no need to multiply anything
            if (!isBias)
            {
                netInput *= getWeightValue(col, row, col + 3, i);
            }
            totalContribution += netInput;
        }
        return totalContribution * outputChangeForNetInputs;
    }

    private void calculateNewWeightsForOutputLayer(double learningRate)
    {
        double[] finalWeightLayer = neuralNetwork[neuralNetwork.length - 3];
        double[] previousInputLayer = neuralNetwork[neuralNetwork.length - 4];
        for (int row = 0; row < finalWeightLayer.length; row++)
        {
            double weight = finalWeightLayer[row];
            double outputNeuronContribution = calculateOutputLayerContribution(row / previousInputLayer.length);
            double previousLayerNeuron = neuralNetwork[neuralNetwork.length - 4][row % previousInputLayer.length];
            double totalContribution = outputNeuronContribution * previousLayerNeuron;
            double newWeight = weight - totalContribution * learningRate;
            newWeights[newWeights.length - 1][row] = newWeight;
        }
    }

    private void calculateNewBiasesForHiddenLayers(double learningRate)
    {
        for (int i=neuralNetwork.length-5; i>=0; i-=3) // for every bias layer but last
        {
            for (int j = 0; j < neuralNetwork[i].length; j++) // for every bias
            {
                double currentBias = neuralNetwork[i][j];
                newBiases[(i-2)/3][j] = currentBias - calculateNeuronContribution(i+1, j, true) * learningRate;
            }
        }
    }

    private void calculateNewBiasesForOutputLayer(double learningRate)
    {
        double[] finalBiasLayer = neuralNetwork[neuralNetwork.length - 2];
        for (int row = 0; row < finalBiasLayer.length; row++)
        {
            double bias = finalBiasLayer[row];
            double totalContribution = calculateOutputLayerContribution(row);
            double newBias = bias - totalContribution * learningRate;
            newBiases[newBiases.length - 1][row] = newBias;
        }
    }

    private double calculateOutputLayerContribution(int neuronIndexRow)
    {
        double neuron = outputLayer[neuronIndexRow];
        double outputChangeForNetInputs = neuron * (1 - neuron);
        double totalErrorChangeForOutput = neuron - targetOutputs[neuronIndexRow];
        return totalErrorChangeForOutput * outputChangeForNetInputs;
    }

    private double getWeightValue(int srcNeuronIndexCol, int srcNeuronIndexRow, int destNeuronIndexCol, int destNeuronIndexRow)
    {
        int[] weightIndex = getWeightIndex(srcNeuronIndexCol, srcNeuronIndexRow, destNeuronIndexCol, destNeuronIndexRow);
        int weightIndexCol = weightIndex[0];
        int weightIndexRow = weightIndex[1];
        return neuralNetwork[weightIndexCol][weightIndexRow];
    }

    private int[] getWeightIndex(int srcNeuronIndexCol, int srcNeuronIndexRow, int destNeuronIndexCol, int destNeuronIndexRow)
    {
        int weightIndexCol = Math.min(srcNeuronIndexCol, destNeuronIndexCol) + 1;
        int weightIndexRow = srcNeuronIndexRow + destNeuronIndexRow * neuralNetwork[weightIndexCol - 1].length;

        return new int[]{weightIndexCol, weightIndexRow};
    }

    private int[] getDestNeuronIndexForWeight(int weightIndexCol, int weightIndexRow)
    {
        int previousInputLayerLength = neuralNetwork[weightIndexCol-1].length;
        int destNeuronIndexCol = weightIndexCol + 2;
        int destNeuronIndexRow = weightIndexRow / previousInputLayerLength;
        return new int[]{destNeuronIndexCol, destNeuronIndexRow};
    }

    private double getDestNeuronValueForWeight(int weightIndexCol, int weightIndexRow)
    {
        int[] destNeuronIndexForWeight = getDestNeuronIndexForWeight(weightIndexCol, weightIndexRow);
        int destNeuronIndexCol = destNeuronIndexForWeight[0];
        int destNeuronIndexRow = destNeuronIndexForWeight[1];
        return neuralNetwork[destNeuronIndexCol][destNeuronIndexRow];
    }

    private int[] getSrcNeuronIndexForWeight(int weightIndexCol, int weightIndexRow)
    {
        int previousInputLayerLength = neuralNetwork[weightIndexCol-1].length;
        int srcNeuronIndexCol = weightIndexCol - 1;
        int srcNeuronIndexRow = weightIndexRow % previousInputLayerLength;
        return new int[]{srcNeuronIndexCol, srcNeuronIndexRow};
    }

    private double getSrcNeuronValueForWeight(int weightIndexCol, int weightIndexRow)
    {
        int[] srcNeuronIndexForWeight = getSrcNeuronIndexForWeight(weightIndexCol, weightIndexRow);
        int srcNeuronIndexCol = srcNeuronIndexForWeight[0];
        int srcNeuronIndexRow = srcNeuronIndexForWeight[1];
        return neuralNetwork[srcNeuronIndexCol][srcNeuronIndexRow];
    }

    private void updateWeightsAndBiases()
    {
        for (int i = 0; i < newWeights.length; i++)
        {
            neuralNetwork[i * 3 + 1] = newWeights[i];
        }
        for (int i = 0; i < newBiases.length; i++)
        {
            neuralNetwork[i * 3 + 2] = newBiases[i];
        }
    }
}
