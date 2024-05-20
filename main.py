import pandas as pd
from sklearn.model_selection import train_test_split

class BreastCancerDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_dataset(self):
        #Reads the dataset from the specified file path and loads it into a pandas DataFrame.
        self.df = pd.read_csv(self.file_path)
        
        return self.df
    
    def clean_data(self):
        #Removes rows with any empty cells from the DataFrame.
        self.df = self.df.dropna()

        return self.df

    def split_data(self):
        #Splits the DataFrame into training (80%) and testing (20%) sets with the random of 42 
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)

        return train_df, test_df

if __name__ == "__main__":
    file_path = 'breast-cancer.csv'
    dataset = BreastCancerDataset(file_path)
    df = dataset.load_dataset()
    print("Initial DF:\n", df.head())
    df_clean = dataset.clean_data()
    train_df, test_df = dataset.split_data()
    
    print("Training DataFrame Head:")
    print(train_df.head())

    print("\nTesting DataFrame Head:")
    print(test_df.head())