#pragma once
#include <dtypes/Tensor.hpp>
#include <string>

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output_gradient, double learning_rate) = 0;
    
    // --- НОВЫЙ МЕТОД ---
    // Устанавливает режим работы слоя (обучение или оценка/вывод)
    // По умолчанию ничего не делает, чтобы не ломать существующие слои.
    virtual void set_training_mode(bool is_training) { 
        (void)is_training; // Подавляем предупреждение о неиспользуемом параметре
    }

    virtual void save_weights(std::ofstream& file) const {};
    virtual void load_weights(std::ifstream& file) {};
};