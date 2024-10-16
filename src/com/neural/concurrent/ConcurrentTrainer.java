package com.neural.concurrent;

import com.neural.DataLoader;
import com.neural.Model;
import com.neural.NetworkTrainer;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ConcurrentTrainer implements NetworkTrainer {
    @Override
    public void train(DataLoader dataLoader, Model model, int epochs, int batchSize, double learningRate)
    {
        int noOfProcessors = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(noOfProcessors);

        ErrorStat errorStat = new ErrorStat();
        Network network = new Network(model, errorStat, epochs, learningRate, batchSize);
        List<Network> networkClones = network.cloneNetwork(noOfProcessors);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            while (dataLoader.hasMoreBatches())
            {
                dataLoader.loadBatch();

                List<Future<?>> futures = new ArrayList<>();
                for (Network networkClone : networkClones)
                {
                    futures.add(executorService.submit(() -> {
                        while (dataLoader.hasNext())
                        {
                            double[][] inputsAndOutputs = dataLoader.getNextDataSample();
                            Model cloneModel = networkClone.getModel();
                            if (inputsAndOutputs != null)
                            {
                                networkClone.incrementDataSampleProcessedCount();
                                cloneModel.setInputs(inputsAndOutputs[0]);
                                cloneModel.setTargetOutputs(inputsAndOutputs[1]);
                                ForwardPropagation.forwardPass(cloneModel);
                                CostCalculator.calculateCost(networkClone);
                                BackwardPropagation.backwardPass(cloneModel);
                            }
                        }
                    }));
                }

                // Wait for all threads to complete
                for (Future<?> future : futures) {
                    try {
                        future.get();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                // Aggregate and average gradients and error
                aggregateAndAverageGradientsAndError(network, networkClones);

                // Display the aggregated error
                displayAggregatedError(network);

                // Clear the error
                network.clearError();

                // Update weights and biases
                network.updateWeightsAndBiases();

                network.clearDataSampleProcessedCount();

                // Clear netNeuronToErrorValues for each clone
                for (Network networkClone : networkClones) {
                    networkClone.clearError();
                    networkClone.clearGradients();
                    networkClone.clearNetNeuronToErrorValues();
                    networkClone.clearDataSampleProcessedCount();
                }
            }
            dataLoader.resetDataSampleIndex();
        }

        executorService.shutdown();
    }

    private void aggregateAndAverageGradientsAndError(Network mainNetwork, List<Network> networkClones) {
        mainNetwork.clearGradients();
        mainNetwork.clearError();

        for (Network clone : networkClones) {
            mainNetwork.addGradients(clone);
            mainNetwork.addError(clone);
            mainNetwork.addDataSampleProcessedCount(clone);
        }

        mainNetwork.averageError();
    }

    private void displayAggregatedError(Network network) {
        System.out.println("Aggregated Error: " + network.getErrorStats().getErrorTotal());
    }
}