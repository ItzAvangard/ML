#pragma once
#include <Layer.hpp>
#include <dtypes/Tensor.hpp>

class SigmoidLayer : public Layer {
private:
    // Кэшируем выход прямого прохода для эффективного вычисления производной
    Tensor last_output;

public:
    SigmoidLayer() = default;

    Tensor forward(const Tensor& input) override;

    Tensor backward(const Tensor& output_gradient, double learning_rate) override;

};