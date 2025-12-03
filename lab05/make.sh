#!/bin/bash
# Build layers
g++ -Wall -Wextra -O2 -std=c++17 -Iinclude -lm -g3 -c src/layers/DenseLayer.cpp -o build/layers/DenseLayer.o
g++ -Wall -Wextra -O2 -std=c++17 -Iinclude -lm -g3 -c src/layers/ConvolutionalLayer.cpp -o build/layers/ConvolutionalLayer.o
g++ -Wall -Wextra -O2 -std=c++17 -Iinclude -lm -g3 -c src/layers/FlattenLayer.cpp -o build/layers/FlattenLayer.o
g++ -Wall -Wextra -O2 -std=c++17 -Iinclude -lm -g3 -c src/layers/MaxPoolingLayer.cpp -o build/layers/MaxPoolingLayer.o
g++ -Wall -Wextra -O2 -std=c++17 -Iinclude -lm -g3 -c src/layers/DropoutLayer.cpp -o build/layers/DropoutLayer.o

# Build activation layers
g++ -Wall -Wextra -O2 -std=c++17 -Iinclude -lm -g3 -c src/layers/activation/SigmoidLayer.cpp -o build/layers/activation/SigmoidLayer.o
g++ -Wall -Wextra -O2 -std=c++17 -Iinclude -lm -g3 -c src/layers/activation/ReLULayer.cpp -o build/layers/activation/ReLULayer.o

# Build model iface class
g++ -Wall -Wextra -O2 -std=c++17 -Iinclude -lm -g3 -c src/Model.cpp -o build/Model.o

# Build main app
g++ -Wall -Wextra -O2 -std=c++17 -Iinclude -lm -g3 -o build/app \
    build/layers/DenseLayer.o \
    build/layers/ConvolutionalLayer.o \
    build/layers/FlattenLayer.o \
    build/layers/MaxPoolingLayer.o \
    build/layers/activation/SigmoidLayer.o \
    build/layers/activation/ReLULayer.o \
    build/layers/DropoutLayer.o \
    build/Model.o \
    src/main.cpp

# Copy built app to root folder
cp build/app app