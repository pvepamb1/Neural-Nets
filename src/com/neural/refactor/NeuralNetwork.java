package com.neural.refactor;

import java.util.Random;

public class NeuralNetwork {

    private int[] layers;
    private double[] inputLayer;
    private double[] outputLayer;
    private double[] targetOutputs;
    private double[][] hiddenLayers;
    private double[][] weights;
    private double[][] biases;
    private double[][] neuralNetwork;

    public NeuralNetwork(int... layers){
       validateLayers(layers);
       initializeLayers(layers);
       assembleLayers();
    }

    private void validateLayers(int... layers){
        if(layers.length < 3){
            throw new IllegalArgumentException("Please specify at least 3 layers");
        }

        for(int layer: layers){
            if(layer <= 0){
                throw new IllegalArgumentException("Please specify only non-zero positive values");
            }
        }
    }

    private void initializeLayers(int... layers){

        this.layers = layers;
        int noOfInputs = layers[0];
        int noOfHiddenLayers = layers.length-2;
        int noOfWeightLayers = noOfHiddenLayers + 1;
        int noOfBiasLayers = noOfHiddenLayers + 1;
        int noOfOutputs = layers[layers.length-1];
        int totalLayers = noOfHiddenLayers + noOfWeightLayers + noOfBiasLayers + 2;

        neuralNetwork = new double[totalLayers][];

        initializeInputLayer(noOfInputs);
        initializeHiddenLayers(noOfHiddenLayers);
        initializeWeights(noOfWeightLayers);
        initializeBiases(noOfBiasLayers);
        //initializeHiddenLayerAndWeights(totalLayers-2);
        initializeOutputLayer(noOfOutputs);
    }

    private void initializeInputLayer(int noOfInputs){
        inputLayer = new double[noOfInputs];
    }

    private void initializeHiddenLayers(int noOfHiddenLayers){
        hiddenLayers = new double[noOfHiddenLayers][];
        for(int i=0; i<noOfHiddenLayers; i++){
            hiddenLayers[i] = new double[layers[i+1]];
        }
    }

    private void initializeWeights(int noOfWeightLayers){
        weights = new double[noOfWeightLayers][];
        for(int i=0; i<noOfWeightLayers; i++){
            weights[i] = getRandomDoubles((long) layers[i] * layers[i+1]);
        }
    }

    private void initializeBiases(int noOfBiasLayers){
        biases = new double[noOfBiasLayers][];
        for(int i=0; i<noOfBiasLayers; i++){
            biases[i] = getRandomDoubles(layers[i+1]);
        }
    }

    /*private void initializeHiddenLayerAndWeights(int noOfHiddenLayersAndWeights){

        for(int i=1, j=0, k=1; i <= noOfHiddenLayersAndWeights; i++){
            if(i%2 != 0){
                neuralNetwork[i] = getRandomDoubles((long) layers[j] * layers[j+1]);
                j++;
            }
            else{
                neuralNetwork[i] = getRandomDoubles(layers[k]);
                k++;
            }
        }
    }*/

    private double[] getRandomDoubles(long streamSize){
        return new Random().doubles(streamSize, 0, 10).toArray();
    }

    private void initializeOutputLayer(int noOfOutputs){
        outputLayer = new double[noOfOutputs];
    }

    private void assembleLayers(){

        neuralNetwork[0] = inputLayer;

        for(int i=0; i<weights.length; i++){
            neuralNetwork[i*3+1] = weights[i];
        }

        for(int i=0; i<biases.length; i++){
            neuralNetwork[i*3+2] = biases[i];
        }

        for(int i=0; i<hiddenLayers.length; i++){
            neuralNetwork[i*3+3] = hiddenLayers[i];
        }

        neuralNetwork[neuralNetwork.length-1] = outputLayer;
    }

    public void setInputs(double[] inputs){
        if(inputs.length != inputLayer.length){
            throw new IllegalArgumentException("Mismatch in specified input size " +
                    "and provided input. Difference: " + (inputs.length - inputLayer.length));
        }
        for (int i = 0; i < inputs.length; i++) {
            neuralNetwork[0] = inputs;
        }
    }

    public void setWeights(double[][] weights){
        if (weights.length != this.weights.length){
            throw new IllegalArgumentException("Incorrect no. of weights provided. Expected: "
                    + this.weights.length + ", provided: " + weights.length);
        }

        for (int i=0; i< weights.length; i++){
            if(this.weights[i].length != weights[i].length){
                throw new IllegalArgumentException("Incorrect no. of weights provided for layer: "
                        + i + ". Expected: " + this.weights.length + ", provided: " + weights.length);
            }
            neuralNetwork[i*3+1] = weights[i];
        }
    }

    public void setBiases(double[][] biases){
        if (biases.length != this.biases.length){
            throw new IllegalArgumentException("Incorrect no. of biases provided. Expected: "
                    + this.biases.length + ", provided: " + biases.length);
        }

        for (int i=0; i< biases.length; i++){
            if(this.biases[i].length != biases[i].length){
                throw new IllegalArgumentException("Incorrect no. of biases provided for layer: "
                        + i + ". Expected: " + this.biases.length + ", provided: " + biases.length);
            }
            neuralNetwork[i*3+2] = biases[i];
        }
    }

    public void train(int iterations, double learningRate){
        forwardPass();
    }

    private void forwardPass(){
        calculateOutput();
        calculateError();
    }

    private void calculateOutput(){

        for(int i=3; i<neuralNetwork.length; i+=3){
            double[] previousInputLayer = neuralNetwork[i-3];
            double[] previousWeightLayer = neuralNetwork[i-2];
            double[] previousBiasLayer = neuralNetwork[i-1];
            for (int j=0; j<neuralNetwork[i].length; j++){
                double totalNetInput = 0;
                for(int k=0; k<previousInputLayer.length; k++){
                    int weightArrayOffset = j * previousInputLayer.length;
                    totalNetInput += previousInputLayer[k] * previousWeightLayer[k+weightArrayOffset]
                            + previousBiasLayer[j];
                }
                totalNetInput = 1 / (1 + Math.exp(-totalNetInput));
                neuralNetwork[i][j] = totalNetInput;
            }
        }
    }

    private void calculateError(){

    }
}
