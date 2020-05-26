#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>
#include <cassert>

#include "neuron.h"

using namespace std;

typedef vector<Neuron> Layer;

class Network{
    public:
        Network(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputs);
        void backProp(const vector<double> &targets);
        void getResults(vector<double> &results) const;
    private:
        vector<Layer> layers;
        double error;
};

#endif