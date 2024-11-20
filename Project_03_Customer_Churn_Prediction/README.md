# Predictive Modeling for Customer Churn Using Deep Learning

## Project Overview
This project involves predicting whether or not a customer is likely to churn using a deep learning model. The task is to choose an appropriate algorithm for predicting customer churn based on the available data, and to evaluate the performance of the model using metrics such as accuracy, precision, recall, and F1 score.

## Key Features
1. **Data Preprocessing**: Load the dataset, handle missing values, and preprocess categorical and numerical features.
2. **Model Building**: Implement a deep learning model using the TensorFlow library.
3. **Training and Evaluation**: Train the model and evaluate its performance using various metrics.
4. **Metrics**: Calculate accuracy, precision, recall, and F1 score to assess model performance.

## Libraries Used
- `pandas`
- `sklearn`
- `tensorflow`
- `keras`

## Code Explanation
- **Data Loading and Preprocessing**: Load the dataset, drop irrelevant columns, convert categorical features to numerical using one-hot encoding, and scale numerical features.
- **Model Building**: Build and compile a deep learning model using dense layers with appropriate activation functions.
- **Model Training**: Train the model using the training dataset.
- **Model Evaluation**: Evaluate the model using accuracy, precision, recall, and F1 score.

## Code Structure
1. Import necessary libraries.
2. Load and preprocess the dataset.
3. Build and compile the deep learning model.
4. Train the model.
5. Evaluate the model and calculate metrics.

## Prerequisites
- Python 3.8+
- Pandas, Scikit-learn, TensorFlow, Keras

## Explanation
This project involves predicting customer churn using a deep learning model built with the TensorFlow library. The dataset contains information about customers, such as their credit score, geography, gender, age, tenure, balance, number of products, and whether they have exited (churned).

- **Data Preprocessing**: The dataset is preprocessed by dropping irrelevant columns, converting categorical features to numerical using one-hot encoding, and scaling numerical features using StandardScaler.
- **Model Building**: A deep learning model is built using dense layers with ReLU activation for hidden layers and sigmoid activation for the output layer.
- **Model Training**: The model is trained using the Adam optimizer and binary cross-entropy loss function.
- **Model Evaluation**: The model's performance is evaluated using accuracy, precision, recall, and F1 score.

## Insights
1. **Model Performance**: The model achieved an accuracy of 86.1%, with a precision of 69.1%, recall of 52.9%, and F1 score of 59.9%.
2. **Feature Importance**: The model can be further improved by exploring feature importance and engineering new features.
3. **Data Preprocessing**: Proper data preprocessing, including scaling and encoding, is crucial for training an effective model.

## Future Enhancements
- **Hyperparameter Tuning**: Experiment with different hyperparameters, such as learning rates, batch sizes, and epochs, to further optimize model performance.
- **Advanced Models**: Implement more advanced models, such as ensemble methods or more complex neural network architectures.
- **Feature Engineering**: Explore and engineer new features to improve model performance and capture more intricate patterns in the data.
