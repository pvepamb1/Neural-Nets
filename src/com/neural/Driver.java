package com.neural;

import com.neural.concurrent.ConcurrentTrainer;
import com.neural.mnist.MnistDataLoader;
import com.neural.mnist.MnistModel;
import com.neural.mnist.MnistNeuralNetwork;
import com.neural.mnist.MnistTester;

import java.util.logging.ConsoleHandler;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Driver
{
    public static void main(String[] args)
    {
        configLog();

        long start = System.currentTimeMillis();

        //NeuralNetwork neuralNetwork = new NeuralNetwork(InputType.CUSTOM, 1,1);
        //neuralNetwork.train(10, 1, 0.11);
        //neuralNetwork.test(new CustomTester());

        DataLoader dataLoader = MnistDataLoader.getInstance();
        MnistDataLoader.loadMnistData("mnistData/train-images.idx3-ubyte","mnistData/train-labels.idx1-ubyte");
        Model model = new MnistModel(32);
        NetworkTrainer networkTrainer = new ConcurrentTrainer();
        networkTrainer.train(dataLoader, model, 100, 32, 0.7);

        MnistDataLoader.loadMnistData("mnistData/t10k-images.idx3-ubyte","mnistData/t10k-labels.idx1-ubyte");
        NetworkTester networkTester = new SimpleTester();
        networkTester.setModel(model);
        networkTester.setTestStrategy(new MnistTester());
        networkTester.setDataLoader(dataLoader);
        networkTester.test();

        /*MnistNeuralNetwork mnistNeuralNetwork = new MnistNeuralNetwork("mnistData/train-images.idx3-ubyte","mnistData/train-labels.idx1-ubyte",  32);
        mnistNeuralNetwork.train(1, 32, 1);
        mnistNeuralNetwork.test("mnistData/t10k-images.idx3-ubyte","mnistData/t10k-labels.idx1-ubyte");*/

        long stop = System.currentTimeMillis();

        printTimeTaken(start, stop);
    }

    private static void configLog()
    {
        Logger rootLogger = Logger.getLogger("");

        // Remove all existing handlers (usually the default ConsoleHandler)
        Handler[] handlers = rootLogger.getHandlers();
        for (Handler handler : handlers)
        {
            rootLogger.removeHandler(handler);
        }

        ConsoleHandler consoleHandler = new ConsoleHandler();
        consoleHandler.setFormatter(new ConsoleLogFormatter());
        consoleHandler.setLevel(Level.INFO);
        rootLogger.addHandler(consoleHandler);
    }

    private static void printTimeTaken(long start, long stop)
    {
        long millis = stop - start;
        long minutes = (millis / 1000)  / 60;
        int seconds = (int)((millis / 1000) % 60);
        System.out.println("Completed in " + minutes + " minutes " + seconds + " seconds and " + millis % 1000 + " milliseconds");
    }
}
