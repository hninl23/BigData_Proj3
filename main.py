# Necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC
import time

class BreastCancerDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self):  # Task 1: Reading the file
        self.df = pd.read_csv(self.file_path)
        return self.df
    
    def clean_data(self):  # Task 2: Data cleaning / Preparation
        self.df = self.df.dropna()
        return self.df

    def split_data(self, target_column):  # Task 2: Data cleaning / Preparation
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_decision_tree(self):  # Task 3: Train and evaluate Decision Tree Classifier
        start_time = time.time()
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        # Visualize the Decision Tree
        plt.figure(figsize=(20,10))
        plot_tree(dt_classifier, filled=True, feature_names=self.X_train.columns, class_names=True)
        plt.show()
        
        # Evaluate the model
        y_pred_dt = dt_classifier.predict(self.X_test)
        accuracy_dt = accuracy_score(self.y_test, y_pred_dt)
        conf_matrix_dt = confusion_matrix(self.y_test, y_pred_dt)
        class_report_dt = classification_report(self.y_test, y_pred_dt, target_names=['Class 0', 'Class 1'],zero_division=0)  # Update target names accordingly
        
        print(f'Training Time: {training_time:.2f} seconds')
        print(f'Accuracy: {accuracy_dt:.2f}')
        print('Confusion Matrix:\n', conf_matrix_dt)
        print('Classification Report:\n', class_report_dt)
        
        # Visualize the confusion matrix
        plt.figure(figsize=(6,6))
        plt.matshow(conf_matrix_dt, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix_dt.shape[0]):
            for j in range(conf_matrix_dt.shape[1]):
                plt.text(x=j, y=i, s=conf_matrix_dt[i, j], va='center', ha='center')
        
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix for Decision Tree')
        plt.show()


    def train_svm(self):  # Task 4: Train and evaluate SVM Classifier
        start_time = time.time()
        svm_classifier = SVC(kernel='rbf', random_state=42)
        svm_classifier.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        

        # Evaluate the model
        y_pred_svm = svm_classifier.predict(self.X_test)
        accuracy_svm = accuracy_score(self.y_test, y_pred_svm)
        conf_matrix_svm = confusion_matrix(self.y_test, y_pred_svm)
        class_report_svm = classification_report(self.y_test, y_pred_svm, target_names=['Class 0', 'Class 1'], zero_division=0)  # Update target names accordingly
        
        print(f'Training Time: {training_time:.2f} seconds')
        print(f'Accuracy: {accuracy_svm:.2f}')
        print('Confusion Matrix:\n', conf_matrix_svm)
        print('Classification Report:\n', class_report_svm)
        
        # Visualize the confusion matrix
        plt.figure(figsize=(6,6))
        plt.matshow(conf_matrix_svm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix_svm.shape[0]):
            for j in range(conf_matrix_svm.shape[1]):
                plt.text(x=j, y=i, s=conf_matrix_svm[i, j], va='center', ha='center')
        
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix for SVM')
        plt.show()


if __name__ == "__main__":
    file_path = 'breast-cancer.csv'
    dataset = BreastCancerDataset(file_path)
    df = dataset.load_dataset()
    print("I am Dataset: \n", df.head)
    df_clean = dataset.clean_data() #Task 2
    print("\nCleaned Data: \n", df_clean)
    target_column = 'diagnosis'
    X_train, X_test, y_train, y_test = dataset.split_data(target_column) #Task 2
    print("\nX_train Data: \n", X_train)
    print("\nX_test Data: \n", X_test )
    print("\ny_train Data: \n", y_train )
    print("\ny_test Data: \n", y_test )

    # Task 3: Train and evaluate Decision Tree Classifier
    dataset.train_decision_tree()

    # Task 4: Train and evaluate SVM Classifier
    dataset.train_svm()


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
