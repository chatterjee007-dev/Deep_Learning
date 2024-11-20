# Handwritten Digit Recognition Using MNIST

## Project Overview
This project involves creating a machine learning model to recognize handwritten digits using the MNIST dataset. The program handles a labeled dataset of 60,000 images and trains multiple models, comparing their accuracy in recognizing handwritten digits. The evaluation is based on metrics like accuracy and F1 score.

## Key Features
1. **Data Preprocessing**: Load, normalize, and reshape the MNIST dataset.
2. **Model Building**: Define and compile a deep learning model using Keras.
3. **Training and Evaluation**: Train the model on the training dataset and evaluate its performance on the test dataset.
4. **Metrics**: Use accuracy and F1 score to evaluate and compare the performance of each model.

## Libraries Used
- `numpy`
- `matplotlib`
- `tensorflow`
- `keras`
- `sklearn`
- `seaborn`

## Code Explanation
- **Data Loading and Preprocessing**: Load the MNIST dataset, normalize pixel values, and reshape images for the model.
- **Model Definition**: Create a Sequential model with dense layers and dropout for regularization.
- **Model Compilation**: Compile the model with appropriate loss function and optimizer.
- **Model Training**: Train the model using the training dataset.
- **Model Evaluation**: Evaluate the model using the test dataset and compute metrics.

## Code Structure
1. Import necessary libraries.
2. Load and preprocess the MNIST dataset.
3. Define and compile the deep learning model.
4. Train the model.
5. Evaluate the model and calculate metrics.

## Prerequisites
- Python 3.8+
- NumPy, Matplotlib, TensorFlow, Keras, Scikit-learn, Seaborn

## Explanation
This project involves training a machine learning model to recognize handwritten digits using the MNIST dataset. The dataset consists of 60,000 28x28 pixel images of handwritten digits for training and 10,000 images for testing. The steps include data preprocessing, model definition, compilation, training, and evaluation.

- **Data Preprocessing**: The MNIST dataset is normalized and reshaped for input into the model. Class labels are converted to one-hot encoded vectors.
- **Model Definition**: A Sequential model is defined with three dense layers and a dropout layer for regularization.
- **Model Compilation**: The model is compiled using the categorical cross-entropy loss function and Adam optimizer.
- **Model Training**: The model is trained on the training dataset for 10 epochs with a batch size of 512.
- **Model Evaluation**: The model is evaluated on the test dataset, achieving an accuracy of 97.42% and a loss of 0.0842.

## Insights
1. **Model Performance**: The model achieved high accuracy (97.42%) in recognizing handwritten digits, indicating its effectiveness.
2. **Data Preprocessing**: Normalizing pixel values and reshaping images were crucial for model training.
3. **Regularization**: Adding a dropout layer helped prevent overfitting and improved model generalization.

## Future Enhancements
- **Hyperparameter Tuning**: Experiment with different hyperparameters, such as learning rates, batch sizes, and epochs, to further optimize model performance.
- **Advanced Models**: Implement more advanced models, such as Convolutional Neural Networks (CNNs), for potentially better performance.
- **Extended Training**: Increase the number of training epochs and use techniques like early stopping to prevent overfitting and achieve better results.
