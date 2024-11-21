# Comparative Analysis of RNN Architectures for Handwritten Digit Recognition

## Project Overview
This project involves implementing and comparing three different Recurrent Neural Network (RNN) architectures using the TensorFlow library. The models are trained and evaluated on the MNIST dataset, ensuring that each model achieves a minimum accuracy of at least 90%.

## Key Features
1. **Three RNN Architectures**: Implement three different RNN architectures (Simple RNN, LSTM, and GRU).
2. **Data Preprocessing**: Normalize pixel values and reshape the dataset for RNN input.
3. **Training and Evaluation**: Train and evaluate the models, ensuring they achieve a minimum accuracy of 90%.
4. **Comparison Table**: Generate a table to compare the accuracy of the different models.

## Libraries Used
- `tensorflow`

## Code Explanation
- **Data Loading and Preprocessing**: Load the MNIST dataset, normalize pixel values, and reshape the data for RNN input.
- **Model Architectures**: Define three different RNN architectures (Simple RNN, LSTM, and GRU).
- **Model Compilation and Training**: Compile and train the models using the categorical cross-entropy loss function and Adam optimizer.
- **Evaluation and Comparison**: Evaluate the models and compare their performance using a comparison table.

## Code Structure
1. Import necessary libraries.
2. Load and preprocess the MNIST dataset.
3. Define three RNN architectures.
4. Compile and train the models.
5. Evaluate the models and create a comparison table.

## Prerequisites
- Python 3.8+
- TensorFlow

## Explanation
This project implements and compares three different Recurrent Neural Network (RNN) architectures using the TensorFlow library. The models are trained on the MNIST dataset, which consists of 60,000 28x28 grayscale images of 10 digits. The three architectures are:

1. **Simple RNN**: A basic RNN architecture with a single SimpleRNN layer.
2. **LSTM**: A more advanced RNN architecture using Long Short-Term Memory (LSTM) units.
3. **GRU**: Another advanced RNN architecture using Gated Recurrent Units (GRU).

The models are compiled using the Adam optimizer and trained with the Categorical Crossentropy loss function. Each model's performance is evaluated and compared based on accuracy.

## Insights
1. **Model Comparison**: The GRU model achieved the highest accuracy, indicating its effectiveness in capturing temporal dependencies in the data.
2. **Training Efficiency**: The LSTM and GRU models showed better performance compared to the Simple RNN, highlighting the advantages of advanced RNN architectures.
3. **Data Preprocessing**: Normalizing pixel values and reshaping input data were crucial steps for training the RNN models effectively.

## Future Enhancements
- **Hyperparameter Tuning**: Experiment with different hyperparameters, such as learning rates, batch sizes, and epochs, to further optimize model performance.
- **Advanced Architectures**: Implement more complex architectures like Bidirectional RNNs or stacked RNNs to improve performance.
- **Extended Training**: Increase the number of training epochs and use techniques like early stopping to prevent overfitting and achieve better results.
