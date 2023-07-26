# Gender-Detector
The code begins by importing the necessary libraries: pandas as pd for data manipulation, CountVectorizer from sklearn.feature_extraction.text for converting text data into numerical vectors, train_test_split from sklearn.model_selection for splitting the dataset into training and testing sets, RandomForestClassifier from sklearn.ensemble for creating a random forest classifier, accuracy_score from sklearn.metrics for evaluating the model's accuracy, and joblib for saving the trained model.

Next, the data is loaded from a CSV file called 'final.csv' using pandas' read_csv function. The 'Name' column is stored in the variable X, and the 'Gender' column is stored in the variable y.

A CountVectorizer object is created to convert the names (X) into numerical vectors (X_vectorized) suitable for machine learning algorithms.

The dataset is split into training and testing sets using train_test_split, with 80% of the data used for training and 20% for testing. The random_state parameter is set to 42 to ensure reproducibility.

A RandomForestClassifier object is instantiated, and the model is trained using the fit method with X_train and y_train.

Predictions are made on the testing set (X_test), and the accuracy of the model is computed using accuracy_score, comparing the predicted labels (y_pred) with the actual labels (y_test). The accuracy is printed to the console.

The code includes a function called predict_gender, which takes a name as input. It vectorizes the name using the pre-trained CountVectorizer object and predicts the gender using the trained RandomForestClassifier. The predicted gender is returned.

The user is prompted to enter a name, which is passed to the predict_gender function. The predicted gender is printed to the console.

Finally, the trained model (classifier) is saved using joblib.dump, and the vocabulary of the vectorizer is saved as well. You can also use pickle to compress it would compress file size more as compared to joblib.

This repository provides a complete example of gender detection using machine learning, allowing users to train their own model and predict the gender of given names.

![Screenshot (1)](https://github.com/AhmedBilalSSG/Gender-Detector/assets/110194946/4d51ff1e-41c3-4137-a95d-88a422a6bac1)
