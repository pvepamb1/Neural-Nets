package com.neural.mnist;

import com.neural.Model;

class MnistModel extends Model
{
    private static final int INPUT_NEURONS = 784; // 28 rows x 28 columns
    private static final int OUTPUT_NEURONS = 10;

    public MnistModel(int... hiddenLayers)
    {
        super(buildLayerInfo(hiddenLayers));
    }


    private static int[] buildLayerInfo(int... hiddenLayers)
    {
        int[] layers = new int[hiddenLayers.length + 2];
        layers[0] = INPUT_NEURONS;
        layers[layers.length - 1] = OUTPUT_NEURONS;
        System.arraycopy(hiddenLayers, 0, layers, 1, hiddenLayers.length); // insert hidden layers in between input and output layers
        return layers;
    }
}
