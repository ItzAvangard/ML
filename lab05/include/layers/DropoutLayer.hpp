// layers/DropoutLayer.hpp
#pragma once
#include <Layer.hpp>
#include <dtypes/Tensor.hpp>

class DropoutLayer : public Layer {
private:
    double dropout_rate;
    bool is_training = false; // По умолчанию слой в режиме оценки
    Tensor dropout_mask; // Маска, чтобы запомнить, какие нейроны были отключены

public:
    explicit DropoutLayer(double rate);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient, double learning_rate) override;
    
    void set_training_mode(bool is_training) override;
};