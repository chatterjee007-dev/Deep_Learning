# Similar Word Finder with NLP

## Project Overview
This project utilizes natural language processing (NLP) techniques to find similar words based on their relationships in a text corpus. The aim is to choose an appropriate algorithm, preprocess the data, and deploy the model on a web browser using Flask for easy accessibility.

## Key Features
1. **Data Preprocessing**: Efficiently handle and preprocess textual data.
2. **Model Building**: Implement and train a Word2Vec model to find word similarities.
3. **Training and Evaluation**: Train the model and evaluate its performance using metrics like cosine similarity.
4. **Deployment**: Deploy the trained model on a web browser using Flask and ngrok.

## Libraries Used
- numpy
- pandas
- seaborn
- matplotlib
- re
- nltk
- gensim
- Flask
- pyngrok

## Code Explanation
- **Data Loading and Preprocessing**: Load and preprocess the text data, including cleaning and tokenization.
- **Model Definition**: Define a Word2Vec model to capture word relationships.
- **Model Compilation**: Compile the model and prepare for training.
- **Model Training**: Train the model on the preprocessed data.
- **Model Evaluation**: Evaluate the model using metrics like cosine similarity.

## Code Structure
1. **Import necessary libraries**
2. **Load and preprocess the dataset**
3. **Define and train the Word2Vec model**
4. **Deploy the model using Flask**
5. **Evaluate the model and test the API**

## Prerequisites
- Python 3.8+
- Google Colab account
- Kaggle account and API key
- Basic understanding of Python, NLP, and Flask

## Explanation
This project involves finding similar words using a Word2Vec model built with Gensim. The steps include data acquisition, preprocessing the text data, defining and training the model, deploying it using Flask, and evaluating its performance.

- **Data Preprocessing**: The dataset is loaded, cleaned, and tokenized. Training and validation datasets are prepared.
- **Model Definition**: A Word2Vec model is defined to capture word relationships.
- **Model Compilation**: The model is compiled and trained using the tokenized text data.
- **Model Training**: The model is trained over multiple epochs to ensure effective learning.
- **Model Evaluation**: The model's performance is evaluated using cosine similarity to ensure accuracy.

## Insights
1. **Model Performance**: The Word2Vec model accurately captures semantic relationships between words.
2. **Data Preprocessing**: Thorough cleaning and preprocessing of text data is crucial for model performance.
3. **Deployment**: Using Flask and pyngrok facilitates easy web deployment and access.

## Future Enhancements
- **Advanced Models**: Implement more sophisticated models like BERT for enhanced performance.
- **Extended Training**: Increase training epochs for improved accuracy and generalization.
- **Hyperparameter Tuning**: Experiment with various hyperparameters to optimize model performance.
