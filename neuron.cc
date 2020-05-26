#include <random>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "neuron.h"

using namespace std;

Neuron::Neuron(unsigned numOutputs, unsigned index){
    for(unsigned connections = 0; connections < numOutputs; connections++){
        output_weights.push_back(Connection());
        output_weights.back().weight = randWeight();
    }
    self_index = index;
    eta = 0.15; // 0 - 1
    alpha = 0.5; // 0 - n
}

double Neuron::getOutputVal() const{
     return output_value;
}

void Neuron::setOutputVal(double newVal){
    output_value = newVal;
}

void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;
    for(unsigned neuro = 0; neuro < prevLayer.size(); neuro++){
        sum += prevLayer[neuro].getOutputVal() * prevLayer[neuro].output_weights[self_index].weight;
    }

    output_value = Neuron::transfer(sum);
}

void Neuron::calcOutputGradient(double target){
    double delta = target - output_value;
    gradient = delta * Neuron::transferDerivative(output_value);
}

void Neuron::calcHiddenGradients(const Layer &layer){
    double dow = sumDOW(layer);
    gradient = dow * Neuron::transferDerivative(output_value);
}

void Neuron::updateInputWeights(Layer &layer){
    for(unsigned neuro = 0; neuro < layer.size(); neuro++){
        Neuron &neuron = layer[neuro];
        double oldDeltaWeight = neuron.output_weights[self_index].deltaWeight;
        double newDeltaWeight = 
                eta //overall learning rate (0.0 slow and 1 super fast)
                * neuron.getOutputVal()
                * gradient
                + alpha //momentum (0.0 none and 0.5 moderate)
                * oldDeltaWeight;
        neuron.output_weights[self_index].deltaWeight = newDeltaWeight;
        neuron.output_weights[self_index].weight += newDeltaWeight;    
    }
}

double Neuron::randWeight(){
    srand(time(NULL));
    return rand() / 1;
}

double Neuron::transfer(double x){
    return tanh(x);
}

double Neuron::transferDerivative(double x){
    return 1.0 - x * x;
}

double Neuron::sumDOW(const Layer &layer)const{
    double sum = 0.0;
    for(unsigned neuro = 0; neuro < layer.size() - 1; neuro++){
        sum += output_weights[neuro].weight * layer[neuro].gradient;
    }

    return sum;
}
