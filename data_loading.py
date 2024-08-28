#importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def dataLoader():
    # Loading a dataset
    # Reading data from a CSV file using the read_csv function and writing it to the df variable
    df = pd.read_csv('toursim_sentiment_analysis_dataset.csv', delimiter=',', encoding='ISO-8859-1')
    # Change the encoding to utf-8
    df = df.applymap(lambda x: str(x).encode("utf-8", errors='surrogatepass').decode("ISO-8859-1", errors='surrogatepass'))

    #splitting the whole data into training and test part in order to train and validate the model performance
    # Perform train_test_split
    x = df['selected_text']  # clean_text
    y = df['sentiment']  # tweet sentiment assessment

    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, train_size=0.8)

    # Label encode the sentiment
    le = LabelEncoder()  # To adapt the target variable for models
    y_train = le.fit_transform(y_train)  # machine learning, transform
    y_test = le.transform(y_test)  # categorical column to numeric

    mapping = list(le.classes_)  # Save the mapping of the classes
    num_classes = len(mapping)

    # Combine features and labels back into DataFrames
    train_df = pd.DataFrame({'selected_text': X_train, 'sentiment': y_train})
    val_df = pd.DataFrame({'selected_text': X_test, 'sentiment': y_test})

    return train_df, val_df, num_classes