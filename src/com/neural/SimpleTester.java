package com.neural;

import static com.neural.concurrent.ForwardPropagation.forwardPass;

public class SimpleTester implements NetworkTester
{
    private TestStrategy testStrategy;
    private Model model;
    private DataLoader dataLoader;

    @Override
    public void setTestStrategy(TestStrategy testStrategy)
    {
        this.testStrategy = testStrategy;
    }

    @Override
    public void setModel(Model model)
    {
        this.model = model;
    }

    @Override
    public void setDataLoader(DataLoader dataLoader)
    {
        this.dataLoader = dataLoader;
    }

    @Override
    public void test()
    {
        dataLoader.resetDataSampleIndex();
        dataLoader.loadBatch();
        while (dataLoader.hasNext())
        {
            double[][] dataSample = dataLoader.getNextDataSample();
            setData(dataSample);
            forwardPass(model);
            testStrategy.apply(model.getOutputLayer(), model.getTargetOutputs(), (int)dataSample[2][0]); // more horrible stuff
            if(!dataLoader.hasNext() && dataLoader.hasMoreBatches())
            {
                dataLoader.loadBatch();
            }
        }
        testStrategy.printResult();
    }

    private void setData(double[][] inputsAndOutputs)
    {
        setInputs(inputsAndOutputs[0]);
        setTargetOutputs(inputsAndOutputs[1]);
    }

    // Todo: The following 4 methods needs to validate the input params. Copy logic from Model class?
    private void setInputs(double[] inputs)
    {
        model.setInputs(inputs);
    }

    private void setTargetOutputs(double[] outputs)
    {
        model.setTargetOutputs(outputs);
    }
}
