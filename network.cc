#include <vector>
#include <iostream>
#include <cassert>

#include "neuron.h"
#include "network.h"

using namespace std;

Network::Network(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; layerNum++){
        layers.push_back(Layer());
        unsigned numOutputs = layerNum == numLayers - 1 ? 0 : topology[layerNum + 1];

        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++){ //adds one additonal neuron as a bias controle
            layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a neuron" << endl;
        }

        layers.back().back().setOutputVal(1.0);
    }
}

void Network::feedForward(const vector<double> &inputs){
    assert(inputs.size() == layers[0].size() - 1);

    for(unsigned in = 0; in < inputs.size(); in++){
        layers[0][in].setOutputVal(inputs[in]);
    }

    for(unsigned layerNum = 1; layerNum < layers.size(); layerNum++){
        Layer &prevLayer = layers[layerNum - 1];
        for(unsigned neuro = 0; neuro < layers[layerNum].size() - 1; neuro++){
            layers[layerNum][neuro].feedForward(prevLayer);
        }
    }
}

void Network::backProp(const vector<double> &targets){
    //calculates overall error of the entire network (RMS) Root Mean Square Error
    Layer &outputLayer = layers.back();
    error = 0.0;
    for(unsigned neuro = 0; neuro < outputLayer.size() - 1; neuro++){
        double delta = targets[neuro] - outputLayer[neuro].getOutputVal();
        error += delta * delta;
    }

    error /= outputLayer.size() - 1;
    error = sqrt(error);

    //calc output layer gradients

    for(unsigned neuro = 0; neuro < outputLayer.size() - 1; neuro++){
        outputLayer[neuro].calcOutputGradient(targets[neuro]);
    }

    //calc gradients on hidden layers
    for(unsigned layerNum = layers.size() - 2; layerNum > 0; layerNum--){
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];

        for(unsigned neuro = 0; neuro < hiddenLayer.size(); neuro++){
            hiddenLayer[neuro].calcHiddenGradients(nextLayer);
        }
    }

    //update connection weights based on those calculations
    for(unsigned layerNum = layers.size() - 1; layerNum > 0; layerNum--){
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];

        for(unsigned neuro = 0; neuro < layer.size() - 1; neuro++){
            layer[neuro].updateInputWeights(prevLayer);
        }
    }
}

void Network::getResults(vector<double> &results) const{
    results.clear();
    for(unsigned neuro = 0; neuro < layers.back().size(); neuro++){
        results.push_back(layers.back()[neuro].getOutputVal());
    }
}