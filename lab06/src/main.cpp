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
    model.train_mode(); // Включаем режим обучения для dropout

    const int epochs = 20;
    const double learning_rate = 0.01;
    model.set_learning_rate(learning_rate);

    std::ofstream training_log("training_log.csv");
    if (!training_log.is_open()) {
        std::cerr << "Warning: Could not open training_log.csv for writing." << std::endl;
    } else {
        training_log << "epoch,training_accuracy,training_loss,validation_accuracy,validation_loss\n";
    }

    try {
        auto all_training_images = read_mnist_images(data_path + "/train-images-idx3-ubyte");
        auto all_training_labels = read_mnist_labels(data_path + "/train-labels-idx1-ubyte");
        std::cout << "Training data loaded: " << all_training_images.size() << " images." << std::endl;

        // Разделяем данные на обучающую и валидационную выборки (автотесты)
        // Берем последние 10% данных для валидации
        const size_t validation_size = all_training_images.size() / 10;
        const size_t train_size = all_training_images.size() - validation_size;
        
        std::vector<std::vector<double>> training_images(
            all_training_images.begin(), 
            all_training_images.begin() + train_size
        );
        std::vector<int> training_labels(
            all_training_labels.begin(), 
            all_training_labels.begin() + train_size
        );
        
        std::vector<std::vector<double>> validation_images(
            all_training_images.begin() + train_size, 
            all_training_images.end()
        );
        std::vector<int> validation_labels(
            all_training_labels.begin() + train_size, 
            all_training_labels.end()
        );
        
        std::cout << "Split data: " << training_images.size() << " training, " 
                  << validation_images.size() << " validation samples." << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::cout << "\n--- Epoch " << epoch + 1 << "/" << epochs << " ---" << std::endl;
            
            // Обучение на обучающей выборке
            model.train_mode(); // Убеждаемся, что модель в режиме обучения
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
            
            // Оценка на обучающей выборке
            model.eval_mode(); // Переключаемся в режим оценки
            int train_correct = 0;
            double train_total_loss = 0.0;
            const size_t train_eval_samples = std::min(static_cast<size_t>(10000), training_images.size());

            for (size_t i = 0; i < train_eval_samples; ++i) {
                Tensor input = Tensor::from_vector(training_images[i], 1, 28, 28);
                Tensor output = model.predict(input);
                
                if (find_predicted_digit(output) == training_labels[i]) {
                    train_correct++;
                }

                std::vector<double> target_vec(10, 0.0);
                target_vec[training_labels[i]] = 1.0;
                for(size_t j = 0; j < output.data.size(); ++j) {
                    train_total_loss += 0.5 * std::pow(target_vec[j] - output.data[j], 2);
                }
            }

            double train_accuracy = static_cast<double>(train_correct) / train_eval_samples * 100.0;
            double train_average_loss = train_total_loss / train_eval_samples;

            // Валидация на обрезанных данных (автотесты)
            int val_correct = 0;
            double val_total_loss = 0.0;
            
            for (size_t i = 0; i < validation_images.size(); ++i) {
                Tensor input = Tensor::from_vector(validation_images[i], 1, 28, 28);
                Tensor output = model.predict(input);
                
                if (find_predicted_digit(output) == validation_labels[i]) {
                    val_correct++;
                }

                std::vector<double> target_vec(10, 0.0);
                target_vec[validation_labels[i]] = 1.0;
                for(size_t j = 0; j < output.data.size(); ++j) {
                    val_total_loss += 0.5 * std::pow(target_vec[j] - output.data[j], 2);
                }
            }

            double val_accuracy = static_cast<double>(val_correct) / validation_images.size() * 100.0;
            double val_average_loss = val_total_loss / validation_images.size();
            
            std::cout << "Epoch " << epoch + 1 
                      << " | Training Accuracy: " << train_accuracy << "%"
                      << " | Training Loss: " << train_average_loss
                      << " | Validation Accuracy: " << val_accuracy << "%"
                      << " | Validation Loss: " << val_average_loss << std::endl;

            if (training_log.is_open()) {
                training_log << epoch + 1 << "," << train_accuracy << "," << train_average_loss 
                            << "," << val_accuracy << "," << val_average_loss << "\n";
            }

            // Сохранение промежуточных результатов на эпохах 5, 10, 15, 20
            if ((epoch + 1) == 5 || (epoch + 1) == 10 || (epoch + 1) == 15 || (epoch + 1) == 20) {
                std::string weights_filename = "cnn_relu_weights_epoch_" + std::to_string(epoch + 1) + ".dat";
                if (model.save_weights(weights_filename)) {
                    std::cout << "Intermediate weights saved to " << weights_filename << std::endl;
                }
            }

            if (target_accuracy > 0 && val_accuracy >= target_accuracy) {
                std::cout << "\nTarget accuracy of " << target_accuracy << "% reached. Stopping training." << std::endl;
                break;
            }
        }

        // Сохраняем финальные веса
        if (model.save_weights("cnn_relu_weights.dat")) {
            std::cout << "\nTraining complete. Final weights saved to cnn_relu_weights.dat" << std::endl;
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