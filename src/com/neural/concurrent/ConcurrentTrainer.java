package com.neural.concurrent;

import com.neural.DataLoader;
import com.neural.Model;
import com.neural.NetworkTrainer;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ConcurrentTrainer implements NetworkTrainer
{

    private final int noOfThreads;
    private static final Logger LOGGER = Logger.getLogger(ConcurrentTrainer.class.getName());

    public ConcurrentTrainer()
    {
        noOfThreads = Runtime.getRuntime().availableProcessors(); // max
    }

    public ConcurrentTrainer(int noOfThreads)
    {
        if(noOfThreads <= 0)
        {
            throw new IllegalArgumentException("Thread count cannot be less than 1");
        }
        this.noOfThreads = noOfThreads;
    }

    @Override
    public void train(DataLoader dataLoader, Model model, int epochs, int batchSize, double learningRate)
    {
        LOGGER.info("Training params: \n\t\tBatch Size: " + batchSize + "\n\t\tLearning Rate: " + learningRate +
                "\n\t\tepochs: " + epochs);
        LOGGER.info(model.toString());

        LOGGER.info("Initializing thread pool with " + noOfThreads + " threads");
        ExecutorService executorService = Executors.newFixedThreadPool(noOfThreads);

        ErrorStat errorStat = new ErrorStat(); // dependency injections, maybe?
        Network network = new Network(model, errorStat, epochs, learningRate, batchSize);
        List<Network> networkClones = network.cloneNetwork(noOfThreads);

        LOGGER.info("Training...");
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            LOGGER.info("Epoch: " + epoch);
            while (dataLoader.hasMoreBatches())
            {
                dataLoader.loadBatch();
                List<Future<?>> futures = new ArrayList<>(); // Generic type?
                for (Network networkClone : networkClones)
                {
                    futures.add(executorService.submit(() ->
                    {
                        while (dataLoader.hasNext()) // Not happy with this level of nesting
                        {
                            double[][] inputsAndOutputs = dataLoader.getNextDataSample();
                            if (inputsAndOutputs != null)
                            {
                                Model modelClone = networkClone.getModel();
                                networkClone.incrementDataSampleProcessedCount();
                                modelClone.setInputs(inputsAndOutputs[0]);
                                modelClone.setTargetOutputs(inputsAndOutputs[1]);
                                ForwardPropagation.forwardPass(modelClone);
                                CostCalculator.calculateCost(networkClone);
                                BackwardPropagation.backwardPass(modelClone);
                            }
                        }
                    }));
                }
                waitForClones(futures);
                aggregateGradientsAndError(network, networkClones);

                network.averageError();
                displayError(network, Level.FINEST);

                network.updateWeightsAndBiases();
                network.clearDataSampleProcessedCount();
                resetClones(networkClones);
            }
            displayError(network, Level.INFO); // Error before update. Rethink later.
            dataLoader.resetDataSampleIndex();
        }
        executorService.shutdown();
    }

    private static void resetClones(List<Network> networkClones)
    {
        for (Network networkClone : networkClones)
        {
            networkClone.clearError();
            networkClone.clearGradients();
            networkClone.clearNetNeuronToErrorValues();
            networkClone.clearDataSampleProcessedCount();
        }
    }

    private void waitForClones(List<Future<?>> futures)
    {
        // Wait for all threads to complete
        for (Future<?> future : futures)
        {
            try
            {
                future.get();
            }
            catch (Exception e)
            {
                LOGGER.severe(e.getMessage());
            }
        }
    }

    private void aggregateGradientsAndError(Network network, List<Network> networkClones)
    {
        network.clearGradients();
        network.clearError();

        for (Network clone : networkClones)
        {
            network.addGradients(clone);
            network.addError(clone);
            network.addDataSampleProcessedCount(clone);
        }
    }

    private void displayError(Network network, Level level)
    {
        String msg = "Error: " + network.getErrorStats().getErrorTotal();

        if ("INFO".matches(level.getName()))
        {
            LOGGER.info(msg);
        }

       LOGGER.finest(msg);
    }
}