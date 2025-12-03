#pragma once
#include <Layer.hpp>

class DenseLayer : public Layer {
private:
    int input_size;
    int output_size;
    Tensor weights; // [output_size x input_size x 1]
    Tensor biases;  // [output_size x 1 x 1]
    Tensor last_input;

public:
    DenseLayer(int input_size, int output_size);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient, double learning_rate) override;
    
    void save_weights(std::ofstream& file) const override;
    void load_weights(std::ifstream& file) override;
};