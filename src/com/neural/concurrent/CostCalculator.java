package com.neural.concurrent;

import com.neural.Model;

public class CostCalculator
{
    public void calculateError(int dataSampleIndex, ErrorStat errorStat, int batchSize)
    {
        double errorTotal = errorStat.getErrorTotal();
        double previousError = errorStat.getPreviousError();
        double minError = errorStat.getMinError();
        double errorRiseFromPreviousCount = errorStat.getErrorRiseFromPreviousCount();
        double errorRiseFromMinCount = errorStat.getErrorRiseFromMinCount();

        // math.min(dataSampleIndex + 1 % batchSize, batchSize) should also work
        double divisor = (dataSampleIndex + 1) % batchSize == 0 ? batchSize : (dataSampleIndex + 1) % batchSize;
        double currentError = errorTotal/divisor;
        System.out.println("Total error: " + currentError);

        // implies that we moved in the wrong direction
        if (currentError > previousError)
        {
            errorRiseFromPreviousCount++;
        }

        // the number of times we missed the global(?) minimum
        if (currentError > minError)
        {
            errorRiseFromMinCount++;
        }
        else
        {
            minError = currentError;
        }

        previousError = currentError;
        errorTotal = 0;
    }

    public static void calculateCost(Network network)
    {
        ErrorStat errorStat = network.getErrorStats();
        Model model = network.getModel();

        double[] outputLayer = model.getOutputLayer();
        double[] targetOutputs = model.getTargetOutputs();
        double errorTotal = errorStat.getErrorTotal();

        double totalError = 0;
        for (int i = 0; i < outputLayer.length; i++)
        {
            totalError += 0.5 * (Math.pow(targetOutputs[i] - outputLayer[i], 2));
        }
        errorTotal += totalError;
        errorStat.setErrorTotal(errorTotal);
    }
}
