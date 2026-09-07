package com.neural.concurrent;

import com.neural.Model;

import java.util.Arrays;

public class BackwardPropagation
{
    public static void backwardPass(Model model)
    {
        int[] layers = model.getLayers();
        double[] inputLayer = model.getInputLayer();
        double[] outputLayer = model.getOutputLayer();
        double[] targetOutputs = model.getTargetOutputs();
        double[][] hiddenLayers = model.getHiddenLayers();
        double[][][] weights = model.getWeights();
        double[][][] weightGradients = model.getWeightGradients();
        double[][] biasGradients = model.getBiasGradients();
        double[][] netNeuronToErrorValues = model.getNetNeuronToErrorValues();

        calculateOutputLayerContribution(outputLayer, targetOutputs, netNeuronToErrorValues);
        calculateWeightGradientsForOutputLayer(layers, inputLayer, hiddenLayers, weights, weightGradients, netNeuronToErrorValues);
        calculateWeightGradientsForHiddenLayers(inputLayer, outputLayer, hiddenLayers, weights, weightGradients, netNeuronToErrorValues);
        calculateBiasGradients(biasGradients, netNeuronToErrorValues);
        clearNetNeuronToErrorValues(netNeuronToErrorValues);
    }

    private static void calculateOutputLayerContribution(double[] outputLayer, double[] targetOutputs,
                                                  double[][] netNeuronToErrorValues)
    {
        for (int i = 0; i < outputLayer.length; i++)
        {
            double neuron = outputLayer[i];
            double outputChangeForNetInputs = neuron * (1 - neuron);
            double totalErrorChangeForOutput = neuron - targetOutputs[i];
            double neuronContribution = totalErrorChangeForOutput * outputChangeForNetInputs;
            netNeuronToErrorValues[netNeuronToErrorValues.length - 1][i] = neuronContribution;
        }
    }

    private static void calculateWeightGradientsForOutputLayer(int[] layers, double[] inputLayer, double[][] hiddenLayers,
                                                        double[][][] weights, double[][][] weightGradients,
                                                        double[][] netNeuronToErrorValues)
    {
        double[][] finalWeightLayer = weights[weights.length - 1];
        double[] previousInputLayer = layers.length < 3 ? inputLayer : hiddenLayers[hiddenLayers.length - 1];
        for (int row = 0; row < finalWeightLayer.length; row++)
        {
            double outputNeuronContribution = netNeuronToErrorValues[netNeuronToErrorValues.length - 1][row];
            for (int col = 0; col < finalWeightLayer[row].length; col++)
            {
                double previousLayerNeuron = previousInputLayer[col];
                double totalContribution = outputNeuronContribution * previousLayerNeuron;
                weightGradients[weightGradients.length - 1][row][col] += totalContribution;
            }
        }
    }

    private static void calculateWeightGradientsForHiddenLayers(double[] inputLayer, double[] outputLayer,
                                                                double[][] hiddenLayers, double[][][] weights,
                                                                double[][][] weightGradients,
                                                                double[][] netNeuronToErrorValues)
    {
        for (int i = weights.length - 2; i >= 0; i--) // for every weight layer but last
        {
            for (int j = 0; j < weights[i].length; j++) // for every weight row (also destNeuronRow)
            {
                for (int k = 0; k < weights[i][j].length; k++) // for every weight (also srcNeuronRow)
                {
                    double[] previousLayer = i != 0 ? hiddenLayers[i - 1] : inputLayer; // Todo: extract to method?
                    double srcNeuron = previousLayer[k];
                    double totalContribution = 0;
                    if (netNeuronToErrorValues[i][j] != 0.0) // the odds the contribution was actually 0?
                    {
                        totalContribution = netNeuronToErrorValues[i][j];
                    }
                    else
                    {
                        double[] destLayer = hiddenLayers[i];
                        double destNeuron = destLayer[j];
                        double outputChangeForNetInputs = destNeuron * (1 - destNeuron);

                        double[] nextLayer = i != hiddenLayers.length - 1 ? hiddenLayers[i + 1] : outputLayer;
                        for (int l = 0; l < nextLayer.length; l++) // 4 nested loops - can we do better?
                        {
                            double nextLayerNeuronContribution = netNeuronToErrorValues[i + 1][l];
                            nextLayerNeuronContribution *= weights[i + 1][l][j]; // too tired to understand index relations
                            totalContribution += nextLayerNeuronContribution;
                        }
                        totalContribution *= outputChangeForNetInputs;
                        netNeuronToErrorValues[i][j] = totalContribution;
                    }
                    weightGradients[i][j][k] += totalContribution * srcNeuron;
                }
            }
        }
    }

    private static void calculateBiasGradients(double[][] biasGradients, double[][] netNeuronToErrorValues)
    {
        for (int i = 0; i < biasGradients.length; i++) // for every bias layer
        {
            for (int j = 0; j < biasGradients[i].length; j++) // for every bias
            {
                biasGradients[i][j] += netNeuronToErrorValues[i][j];
            }
        }
    }

    private static void clearNetNeuronToErrorValues(double[][] netNeuronToErrorValues)
    {
        for (double[] netNeuronToErrorValue : netNeuronToErrorValues)
        {
            Arrays.fill(netNeuronToErrorValue, 0);
        }
    }
}
