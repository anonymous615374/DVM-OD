import pandas as pd
import numpy as np
import time
import os
from Model.DVM_OD import DVM_OD

from sklearn.metrics import (roc_auc_score, precision_score, average_precision_score, recall_score,f1_score, accuracy_score,matthews_corrcoef)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from scipy.spatial.distance import cdist


# Define three arrays of dataset names corresponding to different domains (Classical, NLP, CV)
## You can uncomment the datasets you want to run
dataset_name1 = np.array([  # Classical Datasets
    '1_ALOI'])
    # , '2_annthyroid', '3_backdoor', '4_breastw', '5_campaign',
    # '6_cardio', '7_Cardiotocography', '8_celeba', '9_census', '10_cover', '11_donors', '12_fault', '13_fraud', '14_glass',
    # '15_Hepatitis', '16_http', '17_InternetAds', '18_Ionosphere', '19_landsat', '20_letter',
    # '21_Lymphography', '22_magic.gamma', '23_mammography',
    # '24_mnist', '25_musk', '26_optdigits', '27_PageBlocks',
    # '28_pendigits', '29_Pima', '30_satellite',
    # '31_satimage-2', '32_shuttle','33_skin', '34_smtp', '35_SpamBase',
    # '36_speech', '37_Stamps', '38_thyroid', '39_vertebral',
    # '40_vowels', '41_Waveform', '42_WBC', '43_WDBC', '44_Wilt',
    # '45_wine', '46_WPBC', '47_yeast'])

dataset_name2 = np.array([  # NLP Datasets
    '20news_0'])
    # ,'20news_1','20news_2','20news_3','20news_4','20news_5',
    # 'agnews_0','agnews_1','agnews_2','agnews_3','amazon', 'imdb','yelp'])

dataset_name3 = np.array([  # Computer Vision Datasets
    'CIFAR10_0'])
    # 'CIFAR10_1','CIFAR10_2','CIFAR10_3','CIFAR10_4','CIFAR10_5','CIFAR10_6','CIFAR10_7',
    # 'CIFAR10_8','CIFAR10_9','FashionMNIST_0','FashionMNIST_1','FashionMNIST_2','FashionMNIST_3','FashionMNIST_4','FashionMNIST_5',
    # 'FashionMNIST_6','FashionMNIST_7','FashionMNIST_8','FashionMNIST_9','SVHN_0','SVHN_1','SVHN_2','SVHN_3','SVHN_4',
    # 'SVHN_5','SVHN_6','SVHN_7','SVHN_8','SVHN_9','MNIST-C_brightness', 'MNIST-C_canny_edges','MNIST-C_dotted_line', 'MNIST-C_fog',
    # 'MNIST-C_glass_blur','MNIST-C_identity','MNIST-C_impulse_noise','MNIST-C_motion_blur','MNIST-C_rotate','MNIST-C_scale','MNIST-C_shear',
    # 'MNIST-C_shot_noise','MNIST-C_spatter','MNIST-C_stripe','MNIST-C_translate','MNIST-C_zigzag','MVTec-AD_bottle',
    # 'MVTec-AD_cable','MVTec-AD_capsule','MVTec-AD_carpet','MVTec-AD_grid','MVTec-AD_hazelnut','MVTec-AD_leather',
    # 'MVTec-AD_metal_nut','MVTec-AD_pill','MVTec-AD_screw','MVTec-AD_tile','MVTec-AD_toothbrush',
    # 'MVTec-AD_transistor','MVTec-AD_wood','MVTec-AD_zipper'])

# Define the mapping of domain names to the corresponding dataset folder paths
dataset_links = {
    'Classical': 'Data/Classical/',  # Path to Classical domain datasets
    'NLP': 'Data/NLP_by_BERT/',      # Path to NLP domain datasets
    'CV': 'Data/CV_by_ResNet18/'      # Path to CV domain datasets
}



# Initialize a list to hold the results from dataset processing
data_ans = []

# Define the path to store the final results (CSV format)
output_file = f"Results/DVMOD_result_realworld_data.csv"

# Define the column headers for the output file, including performance metrics
columns = [
    "Dataset", "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 score",
    "PPV (Precision)", "TPR (Recall)", "Time train", "Time test"
]

__SCALER = 'QuantileTransformer'

def distance_vector(point_X, point_Y):
    """
    Calculate pairwise Euclidean distance between two sets of points.
    
    Args:
        point_X (ndarray): Array of shape (N_train, d) where N_train is the number of training samples and d is the number of features.
        point_Y (ndarray): Array of shape (N_test, d) where N_test is the number of test samples and d is the number of features.
        
    Returns:
        ndarray: Distance matrix of shape (N_test, N_train) containing Euclidean distances between each pair of points.
    """
    # Compute squared norms for each point
    norm_X = np.sum(point_X**2, axis=1)  # (N_train,)
    norm_Y = np.sum(point_Y**2, axis=1)  # (N_test,)
    
    # Compute the dot product between the two sets of points
    dot_product = np.dot(point_Y, point_X.T)  # (N_test, N_train)
    
    # Apply Euclidean distance formula
    distance = np.sqrt(abs(norm_Y[:, np.newaxis] + norm_X[np.newaxis, :] - 2 * dot_product))
    return distance

def preprocess_data_OC(train_data, test_data):
    """
    Preprocess the training and testing data by separating features and labels,
    and return the preprocessed feature sets and labels.
    
    Args:
        train_data (DataFrame): Training data containing features and labels.
        test_data (DataFrame): Testing data containing features and labels.
        
    Returns:
        X_train (ndarray): Preprocessed training features.
        y_train (ndarray): Preprocessed training labels.
        X_test (ndarray): Preprocessed testing features.
        y_test (ndarray): Preprocessed testing labels.
    """
    print("..............................Data Overview................................")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)

    # Convert to numpy arrays for easier manipulation
    X_train_total = train_data.iloc[:, :-1].to_numpy()
    y_train_total = train_data.iloc[:, -1].to_numpy()

    X_train = X_train_total[y_train_total == 0]
    y_train = y_train_total[y_train_total == 0]

    print("Train Data Labels [0]:", np.unique(y_train))

    # Prepare test data
    X_test = test_data.iloc[:, :-1].to_numpy()
    y_test = test_data.iloc[:, -1].to_numpy()

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    print("Number of samples:", n_samples)
    print("Number of features:", n_features)
    return X_train, y_train, X_test, y_test


def Model_evaluating(y_true, y_predict, y_scores):
    """
    Evaluate the model using various metrics such as AUC, AUCPR, Accuracy, MCC, F1-score, Precision, and Recall.
    
    Args:
        y_true (ndarray): True labels.
        y_predict (ndarray): Predicted labels.
        y_scores (ndarray): Predicted probabilities for each class.
        
    Returns:
        list: List of evaluation metrics.
    """
    print("..............................Report Parameter...............................")
    
    # Calculate MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_true, y_predict)
    
    # Calculate F1 score
    f1 = f1_score(y_true, y_predict)
    
    # Calculate PPV (Positive Predictive Value) or Precision
    ppv = precision_score(y_true, y_predict)
    
    # Calculate TPR (True Positive Rate) or Recall/Sensitivity
    tpr = recall_score(y_true, y_predict)
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_predict)
    
    # Calculate AUC (Area Under ROC Curve)
    AUC = roc_auc_score(y_true, y_scores[:, 1])
    
    # Calculate AUCPR (Area Under Precision-Recall Curve)
    AUCPR = average_precision_score(y_true, y_scores[:, 1])
    
    # Print out the evaluation metrics
    print("AUCROC:", AUC * 100)
    print("AUCPR:", AUCPR * 100)
    print("Accuracy:", accuracy * 100)
    print("MCC:", mcc)
    print("F1 score:", f1)
    print("PPV (Precision):", ppv)
    print("TPR (Recall):", tpr)
    return [AUC * 100, AUCPR * 100, accuracy * 100, mcc, f1, ppv, tpr]

# Output file initialization (only write header if file doesn't exist)
if not os.path.exists(output_file):
    # Create a CSV file with headers if it doesn't exist
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)

# Function to load and process the dataset (ensure it returns {'X', 'y'})
def load_and_process_dataset(name, domain):
    """
    Load and preprocess a dataset for a given domain and scaler.
    
    Parameters:
    - name (str): The name of the dataset (e.g., '1_ALOI', '20news_0').
    - domain (str): The domain of the dataset (e.g., 'Classical', 'NLP', 'CV').
    
    Returns:
    - data (dict): A dictionary containing 'X' (features) and 'y' (labels) if loading is successful.
    """
    try:
        # Construct the dataset path based on the domain
        dataset_path = f"{dataset_links[domain]}{name}.npz"
        
        # Load the dataset from the path using np.load
        data = np.load(dataset_path, allow_pickle=True)
        
        # Return the data in the form of {'X': X_data, 'y': y_data}
        return data
    except Exception as e:
        # In case of an error during loading, print the error and return None
        print(f"Error loading {domain} dataset {name}: {e}")
        return None
    
def Get_Scaler(name):
  # (StandardScaler, MinMaxScaler, RobustScaler, Normalizer)
  if name == "StandardScaler":
    return StandardScaler()
  if name == "MinMaxScaler":
    return MinMaxScaler()
  if name == "RobustScaler":
    return RobustScaler()
  if name == "Normalizer":
    return Normalizer()
  if name == "QuantileTransformer":
      return QuantileTransformer(output_distribution = "normal", random_state=42)
  if name == "MaxAbsScaler":
      return MaxAbsScaler()
  return None



if __name__ == "__main__":
    # Iterate through datasets
    for dataset_array, domain in zip([dataset_name1, dataset_name2, dataset_name3], ['Classical', 'NLP', 'CV']):
        for name in dataset_array:
            # Load and process the dataset for each dataset name
            data = load_and_process_dataset(name, domain)
            if data is None:
                # Skip the current iteration if data could not be loaded
                continue  

            try:
                # Extract features (X) and labels (y) from the loaded data
                X, y = data['X'], data['y']
                # X, y = pd.DataFrame(X), pd.DataFrame(y)
                print("Original data size:", len(y))

                # Reduce data size if it's too large (only keep 10,000 samples)
                if len(y) > 10000:
                    print("Reducing data size to 10000")
                    # Split data and keep only 10000 samples for training and testing
                    _, X, _, y = train_test_split(X, y, test_size=10000, random_state=42)
                    print("Reduced data size:", len(y))

                # Split data into train/test sets (70% train, 30% test)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                scaler = Get_Scaler(__SCALER)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
                y_train, y_test = pd.DataFrame(y_train), pd.DataFrame(y_test)

                # Combine features and labels for both train and test data
                Train_data = pd.concat([X_train, y_train], axis=1, ignore_index=True)
                Test_data = pd.concat([X_test, y_test], axis=1, ignore_index=True)

                # Track time for model fitting and evaluation
                t0 = time.time()
                # Preprocess data for Outlier Detection (OC) type
                X_train, y_train, X_test, y_test = preprocess_data_OC(Train_data, Test_data)
                
                # Train the model and make predictions
                dvm_od = DVM_OD()
                dvm_od.fit(X_train, y_train)                                                         # Fit the model to the training data
                t1 = time.time()
                y_score = dvm_od.predict(X_test)                                                     # Make predictions on the test data
                t2 = time.time()

                # Transform the training data using the model's transformation
                y_train_score = dvm_od.transform(X_train)

                # Calculate distances for evaluation
                train_score_tmp = distance_vector(y_train_score, y_train_score)
                for i in range(len(train_score_tmp)):
                    train_score_tmp[i , i] = 1e9                                                      # Set diagonal to a very high value to avoid self-distance
                train_score = np.amin(train_score_tmp, axis=1)

                # Calculate probabilities and predictions
                y_proba = np.zeros((len(y_score), 2))
                y_proba[:, 1] = np.minimum(y_score / np.max(train_score), 1)                          # Probability for class 1
                y_proba[:, 0] = 1 - y_proba[:, 1]                                                     # Probability for class 0
                y_predict = (y_proba[:, 1] > 0.5).astype(int)                                         # Convert probabilities to binary predictions

                # Evaluate the model
                v = Model_evaluating(y_test, y_predict, y_proba)

                # Calculate training and testing times
                t_train = t1 - t0
                t_test = t2 - t1
                result = [name] + v + [t_train, t_test]   
                result_df = pd.DataFrame([result], columns=columns)
                result_df.to_csv(output_file, mode='a', header=False, index=False)
                print(f"Result appended to {output_file}")
            except Exception as e:
                print(f"Error processing dataset {name}: {e}")
