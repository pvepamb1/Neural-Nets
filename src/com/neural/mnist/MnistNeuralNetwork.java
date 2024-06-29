package com.neural.mnist;

import com.neural.InputType;
import com.neural.NeuralNetwork;

public class MnistNeuralNetwork extends NeuralNetwork
{
    MnistTester tester;

    public MnistNeuralNetwork(String imgDirPath, String labelDirPath, int... hiddenLayers)
    {
        super(InputType.MNIST, new MnistModel(hiddenLayers));
        MnistDataLoader.loadMnistData(imgDirPath, labelDirPath);
        tester = new MnistTester();
    }

    public void test(String imgDirPath, String labelDirPath)
    {
        MnistDataLoader.loadMnistData(imgDirPath, labelDirPath);
        super.test(tester);
    }

}
