#include <iostream>
#include <string>
#include <vector>
#include <algorithm> // Для std::max_element
#include <iterator>  // Для std::distance

// --- Наши заголовочные файлы ---
#include <Model.hpp>
#include <layers/ConvolutionalLayer.hpp>
#include <layers/MaxPoolingLayer.hpp>
#include <layers/FlattenLayer.hpp>
#include <layers/DenseLayer.hpp>
#include <layers/activation/SigmoidLayer.hpp>
#include <layers/activation/ReLULayer.hpp>
#include <layers/DropoutLayer.hpp>

// --- Вспомогательные заголовочные файлы ---
#include <mnist/mnist_reader.hpp>
#include <mnist/input_loader.hpp>
#include <ext/cxxopts.hpp>


// Вспомогательная функция для определения предсказанной цифры
int find_predicted_digit(const Tensor& output) {
    if (output.data.empty()) return -1;
    auto max_it = std::max_element(output.data.begin(), output.data.end());
    return std::distance(output.data.begin(), max_it);
}

void build_cnn_model(Model& model, double dropout_rate1, double dropout_rate2)  {
    const int input_depth = 1;
    const int num_classes = 10;
    
    model.add(new ConvolutionalLayer(input_depth, 6, 5));
    model.add(new ReLULayer());
    model.add(new MaxPoolingLayer(2));
    model.add(new ConvolutionalLayer(6, 16, 5));
    model.add(new ReLULayer());
    model.add(new MaxPoolingLayer(2));
    model.add(new FlattenLayer());
    model.add(new DenseLayer(256, 120));
    model.add(new ReLULayer());
    if (dropout_rate1 > 0.0) {
        std::cout << "Applying Dropout to 1st Dense layer with rate " << dropout_rate1 << std::endl;
        model.add(new DropoutLayer(dropout_rate1));
    }

    model.add(new DenseLayer(120, 84));
    model.add(new ReLULayer());
    if (dropout_rate2 > 0.0) {
        std::cout << "Applying Dropout to 2nd Dense layer with rate " << dropout_rate2 << std::endl;
        model.add(new DropoutLayer(dropout_rate2));
    }
    model.add(new DenseLayer(84, num_classes));
    model.add(new SigmoidLayer());
}

// Функция для обучения модели
void train_model(const std::string& data_path, double target_accuracy, double dropout1, double dropout2) {
    std::cout << "Starting training with CNN model..." << std::endl;

    Model model;
    build_cnn_model(model, dropout1, dropout2);


    const int epochs = 15;
    const double learning_rate = 0.01;
    model.set_learning_rate(learning_rate);

    std::ofstream training_log("training_log.csv");
    if (!training_log.is_open()) {
        std::cerr << "Warning: Could not open training_log.csv for writing." << std::endl;
    } else {
        training_log << "epoch,accuracy,loss\n";
    }

    try {
        auto training_images = read_mnist_images(data_path + "/train-images-idx3-ubyte");
        auto training_labels = read_mnist_labels(data_path + "/train-labels-idx1-ubyte");
        std::cout << "Training data loaded: " << training_images.size() << " images." << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "\n--- Epoch " << epoch + 1 << "/" << epochs << " ---" << std::endl;
            
            for (size_t i = 0; i < training_images.size(); ++i) {
                Tensor input = Tensor::from_vector(training_images[i], 1, 28, 28);
                std::vector<double> target_vec(10, 0.0);
                target_vec[training_labels[i]] = 1.0;
                Tensor target = Tensor::from_vector(target_vec, 10, 1, 1);
                model.train_step(input, target);

                if ((i + 1) % 10000 == 0) {
                    std::cout << "  Processed " << i + 1 << " / " << training_images.size() << " images." << std::endl;
                }
            }
            
            int correct_predictions = 0;
            double total_loss = 0.0;
            const size_t eval_samples = 10000;

            for (size_t i = 0; i < eval_samples; ++i) {
                Tensor input = Tensor::from_vector(training_images[i], 1, 28, 28);
                Tensor output = model.predict(input);
                
                if (find_predicted_digit(output) == training_labels[i]) {
                    correct_predictions++;
                }

                std::vector<double> target_vec(10, 0.0);
                target_vec[training_labels[i]] = 1.0;
                for(size_t j = 0; j < output.data.size(); ++j) {
                    total_loss += 0.5 * std::pow(target_vec[j] - output.data[j], 2);
                }
            }

            double accuracy = static_cast<double>(correct_predictions) / eval_samples * 100.0;
            double average_loss = total_loss / eval_samples;
            
            std::cout << "Epoch " << epoch + 1 
                      << " | Training Accuracy: " << accuracy << "%"
                      << " | Average Loss: " << average_loss << std::endl;

            if (training_log.is_open()) {
                training_log << epoch + 1 << "," << accuracy << "," << average_loss << "\n";
            }

            if (target_accuracy > 0 && accuracy >= target_accuracy) {
                std::cout << "\nTarget accuracy of " << target_accuracy << "% reached. Stopping training." << std::endl;
                break;
            }
        }

        if (model.save_weights("cnn_relu_weights.dat")) {
            std::cout << "\nTraining complete. Weights saved to cnn_relu_weights.dat" << std::endl;
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
    }
    if (training_log.is_open()) {
        training_log.close();
    }
}

// Функция для тестирования модели
void test_model(const std::string& data_path, double dropout1, double dropout2) {
    std::cout << "Starting testing with CNN model..." << std::endl;
    Model model;
    build_cnn_model(model, dropout1, dropout2);
    model.eval_mode(); 

    if (!model.load_weights("cnn_relu_weights.dat")) {
        std::cerr << "Could not load weights. Train the model first." << std::endl;
        return;
    }
    std::cout << "Weights loaded successfully from cnn_relu_weights.dat." << std::endl;
    
    try {
        auto test_images = read_mnist_images(data_path + "/t10k-images-idx3-ubyte");
        auto test_labels = read_mnist_labels(data_path + "/t10k-labels-idx1-ubyte");
        std::cout << "Test data loaded: " << test_images.size() << " images." << std::endl;
        
        int correct_predictions = 0;
        for (size_t i = 0; i < test_images.size(); ++i) {
            Tensor input = Tensor::from_vector(test_images[i], 1, 28, 28);
            Tensor output = model.predict(input);
            if (find_predicted_digit(output) == test_labels[i]) {
                correct_predictions++;
            }
        }
        double accuracy = static_cast<double>(correct_predictions) / test_images.size() * 100.0;
        std::cout << "\n--- Testing Complete ---" << std::endl;
        std::cout << "Correct Predictions: " << correct_predictions << " / " << test_images.size() << std::endl;
        std::cout << "Accuracy on Test Set: " << accuracy << "%" << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
    }
}

// Функция для вывода на одном изображении
void infer_image(const std::string& image_path, double dropout1, double dropout2) {
    std::cout << "Running inference on " << image_path << std::endl;
    Model model;
    build_cnn_model(model, dropout1, dropout2);
    model.eval_mode();

    if (!model.load_weights("cnn_relu_weights.dat")) {
        std::cerr << "Could not load weights. Train the model first." << std::endl;
        return;
    }

    try {
        auto image_data = load_image(image_path);
        Tensor input = Tensor::from_vector(image_data, 1, 28, 28);
        Tensor output = model.predict(input);
        
        int predicted_digit = find_predicted_digit(output);

        std::cout << "\n--- Inference Complete ---" << std::endl;
        std::cout << "Predicted Digit: " << predicted_digit << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }
}

// Main
int main(int argc, char* argv[]) {
    cxxopts::Options options("CNN MNIST", "Train, test, and infer using a CNN model");

    options.add_options()
        ("t,train", "Train the model (provide path to MNIST data)", cxxopts::value<std::string>())
        ("s,test", "Test the trained model (provide path to MNIST data)", cxxopts::value<std::string>())
        ("i,infer", "Perform inference on a single 28x28 image", cxxopts::value<std::string>())
        ("a,target-accuracy", "Stop training when this accuracy is reached (e.g., 98.5)", cxxopts::value<double>())
        
        // --- НОВЫЕ ФЛАГИ ---
        ("dropout1", "Dropout rate for the 1st dense layer (e.g., 0.5). If > 0, layer is added.", cxxopts::value<double>()->default_value("0.0"))
        ("dropout2", "Dropout rate for the 2nd dense layer (e.g., 0.2). If > 0, layer is added.", cxxopts::value<double>()->default_value("0.0"))
        
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help") || argc == 1) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Получаем значения dropout из аргументов
    double dropout1_rate = result["dropout1"].as<double>();
    double dropout2_rate = result["dropout2"].as<double>();

    if (result.count("train")) {
        double target_accuracy = -1.0;
        if (result.count("target-accuracy")) {
            target_accuracy = result["target-accuracy"].as<double>();
        }
        train_model(result["train"].as<std::string>(), target_accuracy, dropout1_rate, dropout2_rate);
    } else if (result.count("test")) {
        test_model(result["test"].as<std::string>(), dropout1_rate, dropout2_rate);
    } else if (result.count("infer")) {
        infer_image(result["infer"].as<std::string>(), dropout1_rate, dropout2_rate);
    } else {
        std::cout << "Invalid arguments. Use --help for usage." << std::endl;
    }

    return 0;
}