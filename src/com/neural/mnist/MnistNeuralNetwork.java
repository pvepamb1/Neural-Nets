package com.neural.mnist;

import com.neural.InputType;
import com.neural.NeuralNetwork;

public class MnistNeuralNetwork extends NeuralNetwork
{
    MnistTester tester;

    public MnistNeuralNetwork(String imgDirPath, String labelDirPath, int... hiddenLayers)
    {
        super(InputType.MNIST, buildLayerInfo(hiddenLayers));
        MnistDataLoader.loadMnistData(imgDirPath, labelDirPath);
        tester = new MnistTester();
    }

    private static int[] buildLayerInfo(int... hiddenLayers)
    {
        int[] layers = new int[hiddenLayers.length + 2];
        layers[0] = 784; // 28 rows x 28 columns
        layers[layers.length - 1] = 10;
        System.arraycopy(hiddenLayers, 0, layers, 1, hiddenLayers.length);
        return layers;
    }

    public void test(String imgDirPath, String labelDirPath)
    {
        MnistDataLoader.loadMnistData(imgDirPath, labelDirPath);
        super.test(tester);
    }

}
