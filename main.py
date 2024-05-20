# Necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import time

class BreastCancerDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_dataset(self):  # Task 1: Reading the file
        self.df = pd.read_csv(self.file_path)
        return self.df
    
    def clean_data(self):  # Task 2: Data cleaning / Preparation
        self.df = self.df.dropna()
        return self.df

    def split_data(self):  # Task 2: Data cleaning / Preparation
        X = self.df.drop('diagnosis', axis=1)
        y = self.df['diagnosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = 'breast-cancer.csv'
    dataset = BreastCancerDataset(file_path)
    df = dataset.load_dataset()
    df_clean = dataset.clean_data()
    X_train, X_test, y_train, y_test = dataset.split_data()

    # Task 6: Feature importance using Random Forest
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)
    feature_importances = rfc.feature_importances_
    feature_names = X_train.columns

    # Visualize the top two features
    top_two_features = feature_names[feature_importances.argsort()[-2:]]
    plt.scatter(X_train[top_two_features[0]], X_train[top_two_features[1]], c=y_train.map({'M': 1, 'B': 0}))
    plt.xlabel(top_two_features[0])
    plt.ylabel(top_two_features[1])
    plt.show()

    # Remove features with the lowest importance and retrain model
    for num_features_to_remove in [1, 4, 10]:
        features_to_keep = feature_names[feature_importances.argsort()[-(len(feature_names)-num_features_to_remove):]]
        X_train_reduced = X_train[features_to_keep]
        X_test_reduced = X_test[features_to_keep]

        dtc = DecisionTreeClassifier(random_state=42)
        start_time = time.time()
        dtc.fit(X_train_reduced, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time with {len(features_to_keep)} features: {training_time} seconds")

        y_pred_dtc = dtc.predict(X_test_reduced)
        print(f"Decision Tree Classifier Accuracy with {len(features_to_keep)} features:", accuracy_score(y_test, y_pred_dtc))
        print(f"Decision Tree Classifier Recall with {len(features_to_keep)} features:", recall_score(y_test, y_pred_dtc, pos_label='M'))
        print(f"Decision Tree Classifier Confusion Matrix with {len(features_to_keep)} features:\n", confusion_matrix(y_test, y_pred_dtc))

        # Visualize Decision Tree
        plt.figure(figsize=(15,10))
        tree.plot_tree(dtc, filled=True)
        plt.show()
