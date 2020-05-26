#ifndef NEURON_H
#define NEURON_H

#include <random>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron{
    public:
        Neuron(unsigned numOutputs, unsigned index);
        double getOutputVal() const;
        void setOutputVal(double newVal);
        void feedForward(const Layer &prevLayer);

        void calcOutputGradient(double target);
        void calcHiddenGradients(const Layer &layer);
        void updateInputWeights(Layer &layer);
    private:
        static double transfer(double x);
        static double transferDerivative(double x);

        double eta;
        double alpha;
        
        double sumDOW(const Layer &layer)const;
        double randWeight();

        double output_value;
        double gradient;
        unsigned self_index;
        vector<Connection> output_weights;
};

#endif