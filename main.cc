#include <iostream>
#include <vector>
#include <random>
#include <ctime>

#include "network.h"
#include "neuron.h"
using namespace std;

void generateTrainingData(vector<double> &inputs, vector<double> &targets);

int main(){
    vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(3);
    topology.push_back(3);
    topology.push_back(2);

    Network net(topology);

    vector<double> inputs;
    vector<double> targets;
    vector<double> results;

    for(unsigned n = 0; n < 30; n++){
        results.clear();
        generateTrainingData(inputs, targets);
        net.feedForward(inputs);
        net.backProp(targets);
        net.getResults(results);

        cout << "Inputs: " << inputs[1]  << ", " << inputs[1] << endl;

        for(unsigned i = 0; i < results.size(); i++){
            cout << results[i] << endl;
        }
    }
    return 0;
}

void generateTrainingData(vector<double> &inputs, vector<double> &targets){
    srand(time(NULL));
    inputs.clear();
    targets.clear();
    for(unsigned n = 0; n < 2; n++){
        inputs.push_back(rand() % 2);
    }

    (inputs[0] * inputs[1] == 1) ? targets.push_back(1) : targets.push_back(0);
}