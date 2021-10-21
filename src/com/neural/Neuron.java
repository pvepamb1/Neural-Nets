package com.neural;

import java.util.ArrayList;
import java.util.List;


public class Neuron
{
    private List<Connection> connections;
    private OutputWrap output;
    private Float expectedOut;
    private int layer;
    private float bias;

    public Neuron(OutputWrap output, Float expectedOut, int layer, float bias)
    {
        this.output = output;
        this.expectedOut = expectedOut;
        this.layer = layer;
        this.bias = bias;
    }

    public List<Connection> getConnections()
    {
        return connections;
    }

    public void setConnections(List<Connection> connections)
    {
        this.connections = connections;
    }

    public List<Float> getWeights()
    {
        List<Float> temp = new ArrayList<>();
        for (Connection c : connections)
            temp.add(c.getWeight());
        return temp;
    }

    public List<OutputWrap> getInputs()
    {
        List<OutputWrap> temp = new ArrayList<>();
        for (Connection c : connections)
            temp.add(c.getInput().getOutput());
        return temp;
    }

    public OutputWrap getOutput()
    {
        return output;
    }

    public void setOutput(OutputWrap output)
    {
        this.output = output;
    }

    public int getLayer()
    {
        return layer;
    }

    public void setLayer(int layer)
    {
        this.layer = layer;
    }

    public Float getExpectedOut()
    {
        return expectedOut;
    }

    public void setExpectedOut(Float expectedOut)
    {
        this.expectedOut = expectedOut;
    }

    public float getBias()
    {
        return bias;
    }

    public void setBias(float bias)
    {
        this.bias = bias;
    }

}
