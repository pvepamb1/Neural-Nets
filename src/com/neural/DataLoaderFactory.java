package com.neural;

import com.neural.mnist.MnistDataLoader;

public class DataLoaderFactory
{
    public static DataLoader getDataLoader(InputType inputType)
    {
        return switch (inputType)
        {
            case CUSTOM -> new CustomDataLoader();
            case MNIST -> MnistDataLoader.getInstance();
        };
    }
}
