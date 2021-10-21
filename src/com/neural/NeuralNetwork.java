package com.neural;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class NeuralNetwork
{
    private static int hiddenNeurons = 2;
    private static int outputNeurons = 2;
    private static List<Float> inputArray = new ArrayList<>();
    private static List<Neuron> neurons = new ArrayList<>();
    private static List<Connection> weights = new ArrayList<>();
    private static int layer = 1;
    private static float eta = 0.5f;
    private static float[] biasArray = { 0.35f, 0.60f };

    public static void main(String[] args)
    {
        inputArray.add(0.05f);
        inputArray.add(0.10f);

        createNeurons();
        createConnections();
        setNeurons();

        System.out.println("Iterations?");
        Scanner scanner = new Scanner(System.in);
        int iterationCount = scanner.nextInt();
        scanner.close();

        for (int i = 0; i <= iterationCount; i++)
        {
            for (Neuron neuron : neurons)
            {
                if (neuron.getLayer() != 0)
                    calculate(neuron);
            }
            backpropogate();
            update();
        }

        error();
        test();
    }

    public static void createNeurons()
    {
        for (float f : inputArray)
        {
            neurons.add(new Neuron(new OutputWrap(f), null, 0, 0.0f));
        }
        for (int i = 0; i < hiddenNeurons; i++)
        {
            neurons.add(new Neuron(new OutputWrap(0), null, layer, biasArray[0]));
        }
        layer++;
        for (int i = 0; i < outputNeurons; i++)
        {
            neurons.add(new Neuron(new OutputWrap(0), null, layer, biasArray[1]));
        }
    }

    private static void createConnections()
    {
        int id = 0;
        for (int i = 0; i < layer; i++)
        {
            for (Neuron inputNeuron : neurons)
            {
                if (inputNeuron.getLayer() == i)
                {
                    for (Neuron outputNeuron : neurons)
                    {
                        if (outputNeuron.getLayer() == i + 1)
                        {
                            weights.add(new Connection(inputNeuron, outputNeuron, (float) Math.random(), 0.0f, id, i));
                            id++;
                        }
                    }
                }
            }
        }
    }

    public static void setNeurons()
    {
        int i = 0;
        for (Neuron neuron : neurons)
        {
            if (neuron.getLayer() != 0)
            {
                ArrayList<Float> weights = new ArrayList<>();
                ArrayList<OutputWrap> inputs = new ArrayList<>();
                ArrayList<Connection> connections = new ArrayList<>();
                for (Connection c : NeuralNetwork.weights)
                {
                    if (c.getOutput() == neuron)
                    {
                        connections.add(c);
                        weights.add(c.getWeight());
                        inputs.add(c.getInput().getOutput());
                    }
                }
                neuron.setConnections(connections);
            }

            if (neuron.getLayer() == layer)
            {
                switch (i)
                {
                    case 0:
                        neuron.setExpectedOut(0.01f);
                        i++;
                        break;
                    case 1:
                        neuron.setExpectedOut(0.99f);
                        i++;
                        break;
                }
            }
        }
    }

    public static void calculate(Neuron n)
    {
        float total = 0;
        for (int i = 0; i < n.getWeights().size(); i++)
        {
            total += n.getWeights().get(i) * n.getInputs().get(i).getOutput();
        }
        total += n.getBias();
        total = (float) (1 / (1 + Math.exp(-total)));
        n.getOutput().setOutput(total);
    }

    public static void error()
    {
        float total = 0.0f;
        for (Neuron n : neurons)
        {
            if (n.getLayer() == layer)
            {
                total += (float) 1 / 2 * (Math.pow(n.getExpectedOut() - n.getOutput().getOutput(),
                        2));
            }
        }
        System.out.println(total);
    }

    public static void backpropogate()
    {

        int temp = layer - 1;
        while (temp >= 0)
        {
            for (Connection c : weights)
            {
                if (c.getLayer() == temp)
                {
                    if (temp == layer - 1)
                    {
                        float err = c.getOutput().getOutput().getOutput() - c.getOutput()
                                .getExpectedOut();
                        float err2 = c.getOutput().getOutput().getOutput()
                                * (1 - (c.getOutput().getOutput().getOutput()));
                        float totalErr = err * err2 * c.getInput().getOutput().getOutput();
                        c.setNewWeight(c.getWeight() - eta * totalErr);
                    }
                    else
                    {
                        float totalerr1 = 0.0f;
                        Neuron n = c.getOutput();
                        for (Connection c2 : weights)
                        {
                            if (c2.getLayer() == temp + 1)
                            {
                                if (c2.getInput() == n)
                                {
                                    float err = c2.getOutput().getOutput().getOutput()
                                            - c2.getOutput().getExpectedOut();
                                    float err2 = c2.getOutput().getOutput().getOutput()
                                            * (1 - (c2.getOutput().getOutput().getOutput()));
                                    float totalErr = err * err2 * c2.getWeight();
                                    totalerr1 += totalErr;
                                }
                            }
                        }
                        float t2 = c.getOutput().getOutput().getOutput()
                                * (1 - (c.getOutput().getOutput().getOutput()));
                        float t3 = c.getInput().getOutput().getOutput();
                        float t = totalerr1 * t2 * t3;
                        c.setNewWeight(c.getWeight() - eta * t);
                    }
                }
            }
            temp--;
        }
    }

    public static void update()
    {
        for (Connection c : weights)
        {
            c.setWeight(c.getNewWeight());
        }
    }

    public static void test()
    {
        for (Neuron neuron : neurons)
            if (neuron.getLayer() == layer)
                System.out.println(neuron.getOutput().getOutput());

        neurons.get(0).getOutput().setOutput(7.15f);
        neurons.get(1).getOutput().setOutput(8.10f);

        for (Neuron neuron : neurons)
            if (neuron.getLayer() != 0)
                calculate(neuron);

        for (Neuron neuron : neurons)
            if (neuron.getLayer() == layer)
                System.out.println(neuron.getOutput().getOutput());
    }
}
