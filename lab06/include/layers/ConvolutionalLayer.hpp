#pragma once
#include <Layer.hpp>
#include <vector>

class ConvolutionalLayer : public Layer {
private:
    int num_filters;
    int kernel_size;
    Tensor kernels; // Размеры: [num_filters x input_depth x kernel_size x kernel_size]
    Tensor biases;  // Размеры: [num_filters x 1 x 1]
    Tensor last_input;

public:
    ConvolutionalLayer(int input_depth, int num_filters, int kernel_size);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient, double learning_rate) override;
    
    void save_weights(std::ofstream& file) const override;
    void load_weights(std::ifstream& file) override;
};