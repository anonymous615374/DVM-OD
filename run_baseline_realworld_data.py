import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score
)
from pyod.models.ae1svm import AE1SVM
from pyod.models.devnet import DevNet
from pyod.models.lunar import LUNAR
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
from pyod.models.alad import ALAD
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.auto_encoder import AutoEncoder


## You can uncomment the datasets you want to run the experiments on
dataset_name1 = np.array(['1_ALOI'])
    #                       , '2_annthyroid', '3_backdoor', '4_breastw', '5_campaign',
    #    '6_cardio', '7_Cardiotocography', '8_celeba', '9_census', '10_cover', '11_donors', '12_fault', '13_fraud', '14_glass',
    #    '15_Hepatitis', '16_http', '17_InternetAds', '18_Ionosphere', '19_landsat', '20_letter',
    #    '21_Lymphography', '22_magic.gamma', '23_mammography',
    #    '24_mnist', '25_musk', '26_optdigits', '27_PageBlocks',
    #    '28_pendigits', '29_Pima', '30_satellite',
    #    '31_satimage-2', '32_shuttle','33_skin', '34_smtp', '35_SpamBase',
    #    '36_speech', '37_Stamps', '38_thyroid', '39_vertebral',
    #    '40_vowels', '41_Waveform', '42_WBC', '43_WDBC', '44_Wilt',
    #    '45_wine', '46_WPBC', '47_yeast'])

dataset_name2 = np.array([ '20news_0'])
                        #   ,'20news_1','20news_2','20news_3','20news_4','20news_5',
                        #  'agnews_0','agnews_1','agnews_2','agnews_3','amazon', 'imdb','yelp'])

dataset_name3 = np.array([ 'CIFAR10_0'])
        #                   ,'CIFAR10_1','CIFAR10_2','CIFAR10_3','CIFAR10_4','CIFAR10_5','CIFAR10_6','CIFAR10_7',
        # 'CIFAR10_8','CIFAR10_9','FashionMNIST_0','FashionMNIST_1','FashionMNIST_2','FashionMNIST_3','FashionMNIST_4','FashionMNIST_5',
        # 'FashionMNIST_6','FashionMNIST_7','FashionMNIST_8','FashionMNIST_9','SVHN_0','SVHN_1','SVHN_2','SVHN_3','SVHN_4',
        # 'SVHN_5','SVHN_6','SVHN_7','SVHN_8','SVHN_9','MNIST-C_brightness', 'MNIST-C_canny_edges','MNIST-C_dotted_line', 'MNIST-C_fog',
        # 'MNIST-C_glass_blur','MNIST-C_identity','MNIST-C_impulse_noise','MNIST-C_motion_blur','MNIST-C_rotate','MNIST-C_scale','MNIST-C_shear',
        # 'MNIST-C_shot_noise','MNIST-C_spatter','MNIST-C_stripe','MNIST-C_translate','MNIST-C_zigzag','MVTec-AD_bottle',
        # 'MVTec-AD_cable','MVTec-AD_capsule','MVTec-AD_carpet','MVTec-AD_grid','MVTec-AD_hazelnut','MVTec-AD_leather',
        # 'MVTec-AD_metal_nut','MVTec-AD_pill','MVTec-AD_screw','MVTec-AD_tile','MVTec-AD_toothbrush',
        # 'MVTec-AD_transistor','MVTec-AD_wood','MVTec-AD_zipper'])

# Dataset links
dataset_links = {
    'Classical': 'Data/Classical/',
    'NLP': 'Data/NLP_by_BERT/',
    'CV': 'Data/CV_by_ResNet18/'
}

models_list = [
    "CBLOF",
    "KNN",
    "IForest",
    "OCSVM",
    "LOF",
    "DeepSVDD"
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

output_file = f"Results/Baseline_realworld.csv"
columns = ["Dataset", "Model", "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 Score",
           "Precision", "Recall", "Time Train", "Time Test"]


def preprocess_data(train_data, test_data):
    """
    Preprocess the train and test data by separating features and labels,
    and return the preprocessed feature sets and labels.

    Parameters:
    - train_data (pd.DataFrame): The training data which includes both features and labels.
    - test_data (pd.DataFrame): The test data which includes both features and labels.

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

# Define a function to run the model training and evaluation
def run_experiment(dataset_names, dataset_type):
    for name in dataset_names:
        for model_name in models_list:
            try:
                print(f"\nRunning dataset {name} with model {model_name}")

                # Load dataset from the appropriate link
                dataset_path = f"{dataset_links[dataset_type]}{name}.npz"
                if not os.path.exists(dataset_path):
                    print(f"Dataset {name} not found, skipping.")
                    continue

                data = np.load(dataset_path, allow_pickle=True)
                X, y = pd.DataFrame(data['X']), pd.DataFrame(data['y'])

                # Reduce dataset size if too large
                if len(y) > 10000:
                    print("Reducing data size to 10000")
                    _, X, _, y = train_test_split(X, y, test_size=10000, random_state=42)

                # Split into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                
                train_data = pd.concat([X_train, y_train], axis=1, ignore_index=True)
                test_data = pd.concat([X_test, y_test], axis=1, ignore_index=True)

                # Preprocess data (assuming preprocess_data is defined elsewhere)
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

                # Test the model
                start_time = time.time()
                y_pred = model.predict(X_test)

                # Check if the model supports predict_proba and calculate probabilities
                if hasattr(model, "predict_proba"):
                    y_probabilities = model.predict_proba(X_test)  # Class probabilities for each instance
                else:
                    y_probabilities = None  # Some models do not have predict_proba method

                test_time = time.time() - start_time

                # Evaluate the model using the appropriate probabilities
                metrics = evaluate_model(y_test, y_pred, y_scores=None, y_probabilities=y_probabilities)
                result = [name, model_name] + metrics + [train_time, test_time]
                result_df = pd.DataFrame([result], columns=columns)
                result_df.to_csv(output_file, mode='a', header=False, index=False)

                print(f"Results saved for {name} with model {model_name}")

            except Exception as e:
                print(f"Error with dataset {name}, model {model_name}: {e}")



if __name__ == "__main__":
    # Run the experiments for each dataset type
    run_experiment(dataset_name1, 'Classical')
    run_experiment(dataset_name2, 'NLP')
    run_experiment(dataset_name3, 'CV')

