package com.neural;

import com.neural.mnist.MnistDataLoader;

public class DataLoaderFactory
{
    public static DataLoader getDataLoader(InputType inputType)
    {
        return switch (inputType)
        {
            case TEST -> new TestModelLoader();
            case MNIST -> MnistDataLoader.getInstance();
        };
    }
}
