import pandas as pd
import numpy as np
import time
import os 

from Model.DVM_OD import DVM_OD

from sklearn.metrics import (roc_auc_score, precision_score, average_precision_score, recall_score,f1_score, accuracy_score,matthews_corrcoef)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler, QuantileTransformer, RobustScaler, MinMaxScaler, MaxAbsScaler




# Define the three dataset name arrays
## You can uncomment the datasets you want to run the model on
dataset_name = np.array(['1_ALOI'])
                        #  , '2_annthyroid', '3_backdoor', '4_breastw', '5_campaign', '6_cardio', '7_Cardiotocography', '8_celeba', 
                        #  '9_census', '10_cover', '11_donors', '12_fault', '13_fraud', '14_glass', '15_Hepatitis', '16_http', '17_InternetAds',
                        #  '18_Ionosphere', '19_landsat', '20_letter', '21_Lymphography', '22_magic.gamma', '23_mammography', '24_mnist', '25_musk',
                        #  '26_optdigits', '27_PageBlocks', '28_pendigits', '29_Pima', '30_satellite', '31_satimage-2', '32_shuttle', '33_skin', 
                        #  '34_smtp', '35_SpamBase', '36_speech', '37_Stamps', '38_thyroid', '39_vertebral', '40_vowels', '41_Waveform', '42_WBC',
                        #  '43_WDBC', '44_Wilt','20news_0', '20news_1', '20news_2', '20news_3', '20news_4', '20news_5', 'agnews_0', 'agnews_1', 'agnews_2',
                        #  'agnews_3', 'amazon', 'imdb', 'yelp', '45_wine', '46_WPBC', '47_yeast','CIFAR10_0', 'CIFAR10_1', 'CIFAR10_2', 'CIFAR10_3', 'CIFAR10_4',
                        #  'CIFAR10_5','CIFAR10_6','CIFAR10_7', 'CIFAR10_8','CIFAR10_9','FashionMNIST_0','FashionMNIST_1','FashionMNIST_2','FashionMNIST_3','FashionMNIST_4','FashionMNIST_5',
                        #  'FashionMNIST_6','FashionMNIST_7','FashionMNIST_8','FashionMNIST_9','SVHN_0','SVHN_1','SVHN_2','SVHN_3','SVHN_4',
                        #  'SVHN_5','SVHN_6','SVHN_7','SVHN_8','SVHN_9','MNIST-C_brightness', 'MNIST-C_canny_edges','MNIST-C_dotted_line', 'MNIST-C_fog',
                        #  'MNIST-C_glass_blur','MNIST-C_identity','MNIST-C_impulse_noise','MNIST-C_motion_blur','MNIST-C_rotate','MNIST-C_scale','MNIST-C_shear',
                        #  'MNIST-C_shot_noise','MNIST-C_spatter','MNIST-C_stripe','MNIST-C_translate','MNIST-C_zigzag','MVTec-AD_bottle',
                        #  'MVTec-AD_cable','MVTec-AD_capsule','MVTec-AD_carpet','MVTec-AD_grid','MVTec-AD_hazelnut','MVTec-AD_leather',
                        #  'MVTec-AD_metal_nut','MVTec-AD_pill','MVTec-AD_screw','MVTec-AD_tile','MVTec-AD_toothbrush',
                        #  'MVTec-AD_transistor','MVTec-AD_wood','MVTec-AD_zipper'])

## You can change the type of the dataset to run the model on
__TYPE = "global"
__SCALER = "QuantileTransformer"

data_ans = []

output_file = f"Results/DVMOD_result_{__TYPE}_data.csv"

columns = [
    "Dataset", "AUCROC", "AUCPR", "Accuracy", "MCC", "F1 score",
    "PPV (Precision)", "TPR (Recall)", "Time train", "Time test", "Threshold"]


# Define the dataset links (paths to the data)
dataset_links = f'Data/Synthetic_Datasets/{__TYPE}_outliers_datasets/'

# Function to load and process X dataset from CSV
def load_and_process_dataset_X(dataset_name):
    try:
        # Construct the path based on the domain and dataset name
        path = f"{dataset_links}{dataset_name}_X.csv"
        
        # Load the dataset (assuming CSV format)
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Error loading X dataset {dataset_name}: {e}")
        return None

# Function to load and process y dataset from CSV
def load_and_process_dataset_y(dataset_name):
    try:
        # Construct the path based on the domain and dataset name
        path = f"{dataset_links}{dataset_name}_y.csv"
        
        # Load the dataset (assuming CSV format)
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"Error loading y dataset {dataset_name}: {e}")
        return None


# Output file initialization (only write header if file doesn't exist)
if not os.path.exists(output_file):
    pd.DataFrame(columns=columns).to_csv(output_file, index=False)


def distance_vector(null_point_X, null_point_Y):
    # Bình phương các vector hàng
    norm_X = np.sum(null_point_X**2, axis=1)  # (N_train,)
    norm_Y = np.sum(null_point_Y**2, axis=1)  # (N_test,)
    # Tính tích vô hướng
    dot_product = np.dot(null_point_Y, null_point_X.T)  # (N_test, N_train)
    # Áp dụng công thức khoảng cách
    distance = np.sqrt(abs(norm_Y[:, np.newaxis] + norm_X[np.newaxis, :] - 2 * dot_product))
    return distance
    
def preprocess_data_OC(train_data, test_data):
    """
    Preprocess the train and test data by separating features and labels,
    and return the preprocessed feature sets and labels.
    """
    print("..............................Data Overview................................")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)

    # Convert to numpy for easier manipulation
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
    

def Model_evaluating(y_true, y_predict, y_scores ):
    print("..............................Report Parameter...............................")
    
    # Tính MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_true, y_predict)
    
    # Tính F1 score
    f1 = f1_score(y_true, y_predict)
    
    # Tính PPV (Positive Predictive Value) hay còn gọi là Precision
    ppv = precision_score(y_true, y_predict)
    
    # Tính TPR (True Positive Rate) hay còn gọi là Recall hoặc Sensitivity
    tpr = recall_score(y_true, y_predict)
    
    # Tính Accuracy
    accuracy = accuracy_score(y_true, y_predict)
    
    # Tính AUC (Area Under ROC Curve)
    AUC = roc_auc_score(y_true, y_scores[:,1])
    
    # Tính AUCPR (Area Under Precision-Recall Curve)
    AUCPR = average_precision_score(y_true, y_scores[:,1])
    
    # Print out the evaluation metrics
    print("AUCROC:", AUC * 100)
    print("AUCPR:", AUCPR * 100)
    print("Accuracy:", accuracy * 100)
    print("MCC:", mcc)
    print("F1 score:", f1)
    print("PPV (Precision):", ppv)
    print("TPR (Recall):", tpr)
    return [AUC *100, AUCPR * 100, accuracy * 100, mcc, f1, ppv, tpr]
    

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


def get_top_x_percent(y_score, x):
    """
    Hàm lấy phần tử trong top x% của mảng y_score.

    Parameters:
    y_score (array-like): Mảng dữ liệu đầu vào (có thể là một list hoặc một numpy array).
    x (float): Tỷ lệ phần trăm (ví dụ, x = 5 cho top 5%).

    Returns:
    float: Giá trị trong top x% của mảng y_score.
    """
    # Sắp xếp mảng y_score theo thứ tự tăng dần
    sorted_scores = np.sort(y_score)
    
    # Tính chỉ số phần tử tại vị trí top x%
    top_index = int(len(sorted_scores) * ( x / 100))
    
    # Trả về giá trị tại chỉ số đó
    return sorted_scores[top_index]   
    

if __name__ == "__main__":
   # Iterate through datasets
    for dataset_name in dataset_name:
        X = load_and_process_dataset_X(dataset_name)
        y = load_and_process_dataset_y(dataset_name)

        if X is None or y is None:
            continue  # Skip if the dataset couldn't be loaded

        try:
            print("Original data size:", len(y))

            # Split data into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            Train_data = pd.concat([X_train, y_train], axis=1, ignore_index=True)
            Test_data = pd.concat([X_test, y_test], axis=1, ignore_index=True)

            # Track time for model fitting and evaluation
            t0 = time.time()
            
           
            # Apply preprocessing (assuming preprocess_data_OC function exists)
            X_train, y_train, X_test, y_test = preprocess_data_OC(Train_data, Test_data)

            scaler = Get_Scaler(__SCALER)
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Train the model and make predictions
            dvm_od = DVM_OD()
            dvm_od.fit(X_train, y_train)                                                         # Fit the model to the training data
            t1 = time.time()
            y_score = dvm_od.predict(X_test)                                                     # Make predictions on the test data
            t2 = time.time()

            y_train_score = dvm_od.transform(X_train)
            train_score_tmp = distance_vector(y_train_score, y_train_score)
            for i in range(len(train_score_tmp)):
                    train_score_tmp[i , i] = 1e9
            train_score = np.amin(train_score_tmp, axis=1)

            nu = get_top_x_percent(train_score, 99)
            
            
            # Calculate probabilities and predictions
            y_proba = np.zeros((len(y_score), 2))
            y_proba[:, 1] = np.minimum(y_score / nu, 1)  # Probability for class 1
            y_proba[:, 0] = 1 - y_proba[:, 1]           # Probability for class 0
            y_predict = (y_proba[:, 1] > 0.).astype(int)

            # Model evaluation (assuming Model_evaluating function exists)
            v = Model_evaluating(y_test, y_predict, y_proba)

            t_train = t1 - t0
            t_test = t2 - t1

            # Prepare result for appending
            result = [dataset_name] + v + [t_train, t_test] + [nu]
            result_df = pd.DataFrame([result], columns=columns)
            result_df.to_csv(output_file, mode='a', header=False, index=False)
            print(f"Result appended to {output_file}")

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")