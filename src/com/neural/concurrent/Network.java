package com.neural.concurrent;

import com.neural.Model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Network implements Cloneable
{
    private Model model;
    private ErrorStat errorStats;
    private int dataSampleProcessedCount;

    // These should not be in Model but leaving it there for now
    /*private double[] targetOutputs;
    private double[][][] weightGradients;
    private double[][] biasGradients;
    private double[][] netNeuronToErrorValues;*/

    // Training variables
    private double epochs;
    private double learningRate;
    private double batchSize;

    public Network(Model model, ErrorStat errorStats, double epochs, double learningRate, double batchSize)
    {
        this.model = model;
        this.epochs = epochs;
        this.errorStats = errorStats;
        this.learningRate = learningRate;
        this.batchSize = batchSize;
    }

    public void clearGradients()
    {
        for (double[][] weighGradient : model.getWeightGradients())
        {
            for (double[] weight : weighGradient)
            {
                Arrays.fill(weight, 0);
            }
        }

        for (double[] biasGradient : model.getBiasGradients())
        {
            Arrays.fill(biasGradient, 0);
        }
    }

    public void clearNetNeuronToErrorValues()
    {
        for (double[] netNeuronToErrorValue : model.getNetNeuronToErrorValues())
        {
            Arrays.fill(netNeuronToErrorValue, 0);
        }
    }

    public void addGradients(Network clone) {
        // Add gradients from the clone to the main network
        for (int i = 0; i < model.getWeightGradients().length; i++)
        {
            for (int j = 0; j < model.getWeightGradients()[i].length; j++)
            {
                for (int k = 0; k < model.getWeightGradients()[i][j].length; k++)
                {
                    model.getWeightGradients()[i][j][k] += clone.getModel().getWeightGradients()[i][j][k];
                    // divide weightGradients here itself?
                }
            }
        }

        for (int i = 0; i < model.getBiasGradients().length; i++)
        {
            for (int j = 0; j < model.getBiasGradients()[i].length; j++)
            {
                model.getBiasGradients()[i][j] += clone.getModel().getBiasGradients()[i][j];
            }
        }
    }

    public void averageGradients() {
        // Average the gradients by dividing by the number of clones
        for (int i = 0; i < model.getWeightGradients().length; i++)
        {
            for (int j = 0; j < model.getWeightGradients()[i].length; j++)
            {
                for (int k = 0; k < model.getWeightGradients()[i][j].length; k++)
                {
                    model.getWeightGradients()[i][j][k] /= dataSampleProcessedCount;
                }
            }
        }
    }

    public void addDataSampleProcessedCount(Network clone)
    {
        this.dataSampleProcessedCount += clone.getDataSampleProcessedCount();
    }

    public void clearDataSampleProcessedCount()
    {
        setDataSampleProcessedCount(0);
    }

    public void updateWeightsAndBiases() {
        // Update weights and biases using the averaged gradients
        for (int i = 0; i < model.getWeights().length; i++)
        {
            for (int j = 0; j < model.getWeights()[i].length; j++)
            {
                for (int k = 0; k < model.getWeights()[i][j].length; k++)
                {
                    model.getWeights()[i][j][k] -= learningRate * (model.getWeightGradients()[i][j][k]/dataSampleProcessedCount);
                }
            }
        }
        for (int i = 0; i < model.getBiases().length; i++)
        {
            for (int j = 0; j < model.getBiases()[i].length; j++)
            {
                model.getBiases()[i][j] -= learningRate * (model.getBiasGradients()[i][j]/dataSampleProcessedCount);
            }
        }
    }


    public Model getModel()
    {
        return model;
    }

    public void setModel(Model model)
    {
        this.model = model;
    }

    public ErrorStat getErrorStats()
    {
        return errorStats;
    }

    public void setErrorStats(ErrorStat errorStats)
    {
        this.errorStats = errorStats;
    }

    public double getEpochs()
    {
        return epochs;
    }

    public void setEpochs(double epochs)
    {
        this.epochs = epochs;
    }

    public double getLearningRate()
    {
        return learningRate;
    }

    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;
    }

    public double getBatchSize()
    {
        return batchSize;
    }

    public void setBatchSize(double batchSize)
    {
        this.batchSize = batchSize;
    }

    public int getDataSampleProcessedCount()
    {
        return dataSampleProcessedCount;
    }

    public void setDataSampleProcessedCount(int dataSampleProcessedCount)
    {
        this.dataSampleProcessedCount = dataSampleProcessedCount;
    }

    public void incrementDataSampleProcessedCount()
    {
        dataSampleProcessedCount += 1;
    }

    @Override
    public Network clone()
    {
        try
        {
            Network clone = (Network) super.clone();
            clone.model = model.clone();
            clone.errorStats = errorStats.clone();
            return clone;
        }
        catch (CloneNotSupportedException e)
        {
            throw new RuntimeException(e);
        }
    }

    public List<Network> cloneNetwork(int noOfNetworks)
    {
        List<Network> networks = new ArrayList<>();
        for (int i = 0; i < noOfNetworks; i++)
        {
            networks.add(clone());
        }
        return networks;
    }

    public void clearError()
    {
        errorStats.setErrorTotal(0.0d);
    }

    public void addError(Network clone)
    {
        errorStats.setErrorTotal(errorStats.getErrorTotal() + clone.getErrorStats().getErrorTotal());
    }

    public void averageError()
    {
       errorStats.setErrorTotal(errorStats.getErrorTotal() / dataSampleProcessedCount);
    }
}
