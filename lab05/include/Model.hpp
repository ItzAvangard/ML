// Model.hpp
#pragma once
#include "Layer.hpp"
#include <vector>
#include <memory> // Для std::unique_ptr

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    double learning_rate = 0.01;

public:
    Model() = default;

    void add(Layer* layer);
    void set_learning_rate(double rate);

    void train_mode(); // Включить режим обучения
    void eval_mode();  // Включить режим оценки (вывода)
    
    // Прямой проход через всю сеть
    Tensor predict(const Tensor& input);
    
    // Один шаг тренировки: прямой проход, вычисление ошибки и обратный проход
    void train_step(const Tensor& input, const Tensor& target);

    bool save_weights(const std::string& filename);
    bool load_weights(const std::string& filename);
};