// Model.cpp
#include <Model.hpp>
#include <fstream>
#include <iostream>
#include <cmath> // для pow

void Model::add(Layer* layer) {
    layers.emplace_back(layer);
}

void Model::set_learning_rate(double rate) {
    learning_rate = rate;
}

void Model::train_mode() {
    for (auto& layer : layers) {
        layer->set_training_mode(true);
    }
}

void Model::eval_mode() {
    for (auto& layer : layers) {
        layer->set_training_mode(false);
    }
}

Tensor Model::predict(const Tensor& input) {
    Tensor current_output = input;
    for (const auto& layer : layers) {
        current_output = layer->forward(current_output);
    }
    return current_output;
}

void Model::train_step(const Tensor& input, const Tensor& target) {
    // 1. Прямой проход
    Tensor output = predict(input);

    // 2. Вычисление градиента ошибки (для MSE - Mean Squared Error)
    if (output.data.size() != target.data.size()) {
        throw std::runtime_error("Output and target sizes mismatch for loss calculation.");
    }
    Tensor loss_gradient(output.depth, output.height, output.width);
    for (size_t i = 0; i < output.data.size(); ++i) {
        loss_gradient.data[i] = output.data[i] - target.data[i];
    }

    // 3. Обратный проход
    Tensor current_gradient = loss_gradient;
    for (int i = layers.size() - 1; i >= 0; --i) {
        current_gradient = layers[i]->backward(current_gradient, learning_rate);
    }
}

bool Model::save_weights(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }
    for (const auto& layer : layers) {
        layer->save_weights(file);
    }
    return true;
}

bool Model::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
     if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return false;
    }
    for (auto& layer : layers) {
        layer->load_weights(file);
    }
    return true;
}