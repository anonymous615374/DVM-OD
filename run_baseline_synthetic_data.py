import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import KernelCenterer,LabelEncoder, MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.cof import COF
from pyod.models.loda import LODA
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.alad import ALAD
from pyod.models.ae1svm import AE1SVM
from pyod.models.devnet import DevNet
from pyod.models.lunar import LUNAR

## You can uncomment the following datasets to run the experiments on all the datasets
dataset_names = np.array(['1_ALOI'])
                        #   , '2_annthyroid', '3_backdoor', '4_breastw', '5_campaign', '6_cardio', '7_Cardiotocography', '8_celeba', 
                        #  '9_census', '10_cover', '11_donors', '12_fault', '13_fraud', '14_glass', '15_Hepatitis', '16_http', '17_InternetAds',
                        #  '18_Ionosphere', '19_landsat', '20_letter', '21_Lymphography', '22_magic.gamma', '23_mammography', '24_mnist', '25_musk',
                        #  '26_optdigits', '27_PageBlocks', '28_pendigits', '29_Pima', '30_satellite', '31_satimage-2', '32_shuttle', '33_skin', 
                        #  '34_smtp', '35_SpamBase', '36_speech', '37_Stamps', '38_thyroid', '39_vertebral', '40_vowels', '41_Waveform', '42_WBC',
                        #  '43_WDBC', '44_Wilt', '45_wine', '46_WPBC', '47_yeast'])
models_list = [
    "CBLOF",
    "KNN",
    "IForest",
    "OCSVM",
    "LOF",
    "DeepSVDD",
    "HBOS",
    "PCA",
    "SOD",
    "COF",
    "LODA",
    "ECOD",
    "COPOD",
    "AutoEncoder",
     "DevNet",
    "LUNAR",
    "AE1SVM",
    "ALAD"
]

__TYPE = "global"
output_file = f"Results/Baseline_{__TYPE}.csv"
columns = ["Dataset", "Model", "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 Score",
           "Precision", "Recall", "Time Train", "Time Test"]



def load_and_process_dataset_X(dataset_links, dataset_name):
    """
    Loads the features (X) dataset from a CSV file.

    Parameters:
    - dataset_links (str): The directory path where the dataset files are stored.
    - dataset_name (str): The name of the dataset (excluding the suffix).

    Returns:
    - pd.DataFrame: The features (X) dataset as a Pandas DataFrame, or None if an error occurs.
    """
    try:
        # Construct the path based on the domain and dataset name
        path = os.path.join(dataset_links, f"{dataset_name}_X.csv")
        
        # Load the dataset (assuming CSV format)
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Error loading X dataset {dataset_name}: {e}")
        return None

# Function to load and process y dataset from CSV
def load_and_process_dataset_y(dataset_links, dataset_name):
    """
    Loads the labels (y) dataset from a CSV file.

    Parameters:
    - dataset_links (str): The directory path where the dataset files are stored.
    - dataset_name (str): The name of the dataset (excluding the suffix).

    Returns:
    - pd.DataFrame: The labels (y) dataset as a Pandas DataFrame, or None if an error occurs.
    """
    try:
        # Construct the path based on the domain and dataset name
        path = os.path.join(dataset_links, f"{dataset_name}_y.csv")
        
        # Load the dataset (assuming CSV format)
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Error loading y dataset {dataset_name}: {e}")
        return None

# Function to preprocess the data (split into features and labels)
def preprocess_data(train_data, test_data):
    """
    Preprocesses the training and test data by separating features (X) and labels (y).
    
    Parameters:
    - train_data (pd.DataFrame): The training data including both features and labels.
    - test_data (pd.DataFrame): The test data including both features and labels.

    Returns:
    - X_train (np.ndarray): The feature set for training data (excluding the labels).
    - y_train (np.ndarray): The label set for training data (the last column).
    - X_test (np.ndarray): The feature set for test data (excluding the labels).
    - y_test (np.ndarray): The label set for test data (the last column).
    """
    print("..............................Data Overview................................")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)

    # Convert to numpy for easier manipulation
    X_train_total = train_data.iloc[:, :-1].to_numpy()  # Extract features (all columns except last)
    y_train_total = train_data.iloc[:, -1].to_numpy()   # Extract labels (last column)

    # Filter training data for the class label 0 (normal class)
    X_train = X_train_total[y_train_total == 0]
    y_train = y_train_total[y_train_total == 0]

    print("Train Data Labels [0]:", np.unique(y_train))  # Print the unique labels for training data

    # Prepare test data
    X_test = test_data.iloc[:, :-1].to_numpy()  # Extract features from test data
    y_test = test_data.iloc[:, -1].to_numpy()   # Extract labels from test data

    # Display the number of samples and features in training data
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    print("Number of samples:", n_samples)
    print("Number of features:", n_features)

    # Return the preprocessed data
    return X_train, y_train, X_test, y_test

# Function to evaluate the model using various metrics
def evaluate_model(y_true, y_pred, y_scores=None, y_probabilities=None):
    """
    Evaluates the model using multiple metrics and prints the results.
    This version includes AUCROC and AUCPR using predicted probabilities.

    Parameters:
    - y_true (np.ndarray): True labels of the test data.
    - y_pred (np.ndarray): Predicted labels of the test data.
    - y_scores (np.ndarray, optional): Scores used to compute AUCROC and AUCPR.
    - y_probabilities (np.ndarray, optional): Predicted probabilities for each class.

    Returns:
    - metrics (list): A list containing AUCROC, AUCPR, accuracy, MCC, F1 score, precision, and recall.
    """
    print("..............................Evaluation Metrics...............................")

    # Calculate standard metrics
    mcc = matthews_corrcoef(y_true, y_pred)  # Matthew's correlation coefficient
    f1 = f1_score(y_true, y_pred)  # F1 score
    precision = precision_score(y_true, y_pred)  # Precision
    recall = recall_score(y_true, y_pred)  # Recall
    accuracy = accuracy_score(y_true, y_pred)  # Accuracy

    # ROC and PR curve scores using predicted probabilities
    auc_roc, auc_pr = None, None
    if y_probabilities is not None:
        auc_roc = roc_auc_score(y_true, y_probabilities[:, 1])  # Probabilities for class 1 (outliers)
        auc_pr = average_precision_score(y_true, y_probabilities[:, 1])  # AUCPR for class 1

    # Display metrics
    print(f"AUCROC: {auc_roc * 100 if auc_roc else 'N/A'}")
    print(f"AUCPR: {auc_pr * 100 if auc_pr else 'N/A'}")
    print(f"Accuracy: {accuracy * 100:.2f}")
    print(f"MCC: {mcc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Return the evaluation metrics as a list
    return [auc_roc * 100 if auc_roc else None, auc_pr * 100 if auc_pr else None,
            accuracy * 100, mcc, f1, precision, recall]

# Function to return the appropriate model based on the name
def get_model(name, **kwargs):
    """
    Returns the appropriate PyOD model based on the provided name.
    
    Parameters:
    - name (str): The name of the anomaly detection model (e.g., 'CBLOF', 'KNN').
    - kwargs (dict): Additional parameters to initialize the model.

    Returns:
    - model (object): An instance of the specified anomaly detection model.
    """
    model_dict = {
        "CBLOF": CBLOF,
        "KNN": KNN,
        "IForest": IForest,
        "OCSVM": OCSVM,
        "LOF": LOF,
        "DeepSVDD": DeepSVDD,
        "HBOS": HBOS,
        "PCA": PCA,
        "SOD": SOD,
        "COF": COF,
        "LODA": LODA,
        "ECOD": ECOD,
        "COPOD": COPOD,
        "AutoEncoder": AutoEncoder,
        "DevNet": DevNet,
        "LUNAR": LUNAR,
        "AE1SVM": AE1SVM,
        "ALAD": ALAD
    }
    
    # Fetch the model class based on the provided model name
    model_class = model_dict.get(name)
    
    # Raise an error if the model name is not found
    if model_class is None:
        raise ValueError(f"Model {name} not found.")
    
    # Return the model initialized with the provided parameters
    return model_class(**kwargs)




# Initialize output CSV file
if not os.path.exists(output_file):
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)


if __name__ == "__main__":
    for name in dataset_names:
        for model_name in models_list:
            try:
                dataset_links = f'Data/Synthetic_Datasets/{__TYPE}_outliers_datasets/'
                print(f"\nRunning dataset {name} with model {model_name}")

                X = load_and_process_dataset_X(dataset_links, name)
                y = load_and_process_dataset_y(dataset_links, name)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                train_data = pd.concat([X_train, y_train], axis=1, ignore_index=True)
                test_data = pd.concat([X_test, y_test], axis=1, ignore_index=True)

                X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)

                if model_name == 'DeepSVDD':
                    start_time = time.time()
                    n_features = X_train.shape[1]
                    model = DeepSVDD(n_features=n_features)
                    model.fit(X_train)
                else:
                    model = get_model(model_name)
                    start_time = time.time()
                    if model_name == 'DevNet':
                        model.fit(X_train, y_train)
                    else:
                        model.fit(X_train)
                        
                train_time = time.time() - start_time

                y_pred = model.predict(X_test)

                if hasattr(model, "predict_proba"):
                    y_probabilities = model.predict_proba(X_test)
                else:
                    y_probabilities = None

                test_time = time.time() - start_time

                metrics = evaluate_model(y_test, y_pred, y_scores=None, y_probabilities=y_probabilities)
                result = [name, model_name] + metrics + [train_time, test_time]
                result_df = pd.DataFrame([result], columns=columns)
                result_df.to_csv(output_file, mode='a', header=False, index=False)

                print(f"Results saved for {name} with model {model_name}")

            except Exception as e:
                print(f"Error with dataset {name}, model {model_name}: {e}")