import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE
import numpy as np
from sklearn.mixture import GaussianMixture


output_folder_cluster = 'Data/Synthetic_Datasets/cluster_outliers_datasets/'  # Folder for cluster data
output_folder_local = 'Data/Synthetic_Datasets/local_outliers_datasets/'
output_folder_global = 'Data/Synthetic_Datasets/global_outliers_datasets/'
output_folder_dependency = 'Data/Synthetic_Datasets/dependency_outliers_datasets/'

os.makedirs(output_folder_cluster, exist_ok=True)
os.makedirs(output_folder_local, exist_ok=True)
os.makedirs(output_folder_global, exist_ok=True)
os.makedirs(output_folder_dependency, exist_ok=True)



# Dataset links
dataset_links = {
    'Classical': 'Data/Classical/',
    'NLP': 'Data/NLP_by_BERT/',
    'CV': 'Data/CV_by_ResNet18/'
}

## You can uncomment the datasets you want to generate synthetic data for
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

## You can change the type of anomaly here: local, global, cluster, dependency
__TYPES = "global"


def generate_realistic_synthetic(X, y, realistic_synthetic_mode, alpha:int, percentage:float, seed:int=42):
    '''
    Currently, four types of realistic synthetic outliers can be generated:
    1. local outliers: where normal data follows the GMM distribution, and anomalies follow the GMM distribution with modified covariance
    2. global outliers: where normal data follows the GMM distribution, and anomalies follow the uniform distribution
    3. dependency outliers: where normal data follows the vine copula distribution, and anomalies follow the independent distribution captured by GaussianKDE
    4. cluster outliers: where normal data follows the GMM distribution, and anomalies follow the GMM distribution with modified mean

    :param X: input X
    :param y: input y
    :param realistic_synthetic_mode: the type of generated outliers
    :param alpha: the scaling parameter for controlling the generated local and cluster anomalies
    :param percentage: controlling the generated global anomalies
    :param seed: random seed for reproducibility
    '''

    if realistic_synthetic_mode not in ['local', 'cluster', 'dependency', 'global']:
        raise NotImplementedError(f"Mode {realistic_synthetic_mode} is not implemented.")

    # the number of normal data and anomalies
    pts_n = len(np.where(y == 0)[0])
    pts_a = len(np.where(y == 1)[0])

    # only use the normal data to fit the model
    X_normal = X[y.values == 0]
    y_normal = y[y == 0]

    # generate the synthetic normal data
    if realistic_synthetic_mode in ['local', 'cluster', 'global']:
        # select the best n_components based on the BIC value
        metric_list = []
        n_components_list = list(np.arange(1, 10))

        for n_components in n_components_list:
            gm = GaussianMixture(n_components=n_components, random_state=seed).fit(X_normal)
            metric_list.append(gm.bic(X_normal))

        best_n_components = n_components_list[np.argmin(metric_list)]

        # refit based on the best n_components
        gm = GaussianMixture(n_components=best_n_components, random_state=seed).fit(X_normal)

        # generate the synthetic normal data
        X_synthetic_normal = gm.sample(pts_n)[0]

    elif realistic_synthetic_mode == 'dependency':
        # sampling the feature since copulas method may spend too long to fit
        if X.shape[1] > 50:
            idx = np.random.choice(np.arange(X.shape[1]), 50, replace=False)
            X_normal = X_normal[:, idx]
      
        copula = VineCopula('center')  # default is the C-vine copula
        if X_normal.shape[0] > 2000:
            X_sampled = X_normal.sample(n=2000, random_state=42)  # Sample 2000 rows
        else:
            X_sampled = X_normal  # Use all rows if there are fewer than 2000
        copula.fit(pd.DataFrame(X_sampled))
       
        # sample to generate synthetic normal data
        X_synthetic_normal = copula.sample(pts_n).values
        print(X_synthetic_normal)
       
    else:
        pass

    # generate the synthetic abnormal data
    if realistic_synthetic_mode == 'local':
        # generate the synthetic anomalies (local outliers)
        gm.covariances_ = alpha * gm.covariances_
        X_synthetic_anomalies = gm.sample(pts_a)[0]

    elif realistic_synthetic_mode == 'cluster':
        # generate the clustering synthetic anomalies
        gm.means_ = alpha * gm.means_
        X_synthetic_anomalies = gm.sample(pts_a)[0]

    elif realistic_synthetic_mode == 'dependency':
        print("helo")
        X_synthetic_anomalies = np.zeros((pts_a, X_normal.shape[1]))
        print("helo")
        
        # using the GaussianKDE for generating independent features
        for i in range(X_normal.shape[1]):
            kde = GaussianKDE()
            kde.fit(X_normal.iloc[:, i])  # Use .iloc to index columns by position
            X_synthetic_anomalies[:, i] = kde.sample(pts_a)
        print("helo")
    elif realistic_synthetic_mode == 'global':
        # generate the synthetic anomalies (global outliers)
        X_synthetic_anomalies = []

        for i in range(X_synthetic_normal.shape[1]):
            low = np.min(X_synthetic_normal[:, i]) * (1 + percentage)
            high = np.max(X_synthetic_normal[:, i]) * (1 + percentage)

            X_synthetic_anomalies.append(np.random.uniform(low=low, high=high, size=pts_a))

        X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

    else:
        pass

    # Concatenate normal and anomalous data
    X_combined = np.concatenate((X_synthetic_normal, X_synthetic_anomalies), axis=0)
    y_combined = np.append(np.repeat(0, X_synthetic_normal.shape[0]),
                           np.repeat(1, X_synthetic_anomalies.shape[0]))

    return X_combined, y_combined



# Function to load and process the dataset
def load_and_process_dataset(name, domain):
    try:
        dataset_path = f"{dataset_links[domain]}{name}.npz"
        data = np.load(dataset_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading {domain} dataset {name}: {e}")
        return None
    

if __name__ == "__main__":
    # Iterate over datasets and generate synthetic data
    for dataset_array, domain in zip([dataset_name1, dataset_name2, dataset_name3], ['Classical', 'NLP', 'CV']):
        for name in dataset_array:
            data = load_and_process_dataset(name, domain)
            if data is None:
                continue  # Skip if dataset couldn't be loaded

            try:
                X, y = data['X'], data['y']
                X, y = pd.DataFrame(X), pd.DataFrame(y)
                print("Original data size:", len(y))
        
                # Reduce data size if too large
                if len(y) > 10000:
                    print("Reducing data size to 10000")
                    _, X, _, y = train_test_split(X, y, test_size=10000, random_state=42)
        
                # Generate synthetic data for data __TYPES
                if __TYPES == 'global':
                    X_gen, y_gen = generate_realistic_synthetic(X, y, "global", alpha=1.1, percentage=0.1)
                if __TYPES == 'cluster':
                    X_gen, y_gen = generate_realistic_synthetic(X, y, "cluster", alpha=5, percentage=0.1)
                if __TYPES == 'local':
                    X_gen, y_gen = generate_realistic_synthetic(X, y, "local", alpha=5, percentage=0.1)
                if __TYPES == 'dependency':
                    X_gen, y_gen = generate_realistic_synthetic(X, y, "dependency", alpha=1.1, percentage=0.1)
                
        
                # Convert the synthetic data to DataFrame before saving
                X_gen_df = pd.DataFrame(X_gen)
                y_gen_df = pd.DataFrame(y_gen)
        
                # Save synthetic data for data __TYPES
                if __TYPES == 'global':
                    output_filename_X= os.path.join(output_folder_global, f'{name}_X.csv')
                    output_filename_y = os.path.join(output_folder_global, f'{name}_y.csv')
                if __TYPES == 'cluster':
                    output_filename_X= os.path.join(output_folder_cluster, f'{name}_X.csv')
                    output_filename_y = os.path.join(output_folder_cluster, f'{name}_y.csv')
                if __TYPES == 'local':
                    output_filename_X= os.path.join(output_folder_local, f'{name}_X.csv')
                    output_filename_y = os.path.join(output_folder_local, f'{name}_y.csv')
                if __TYPES == 'dependency':
                    output_filename_X= os.path.join(output_folder_dependency, f'{name}_X.csv')
                    output_filename_y = os.path.join(output_folder_dependency, f'{name}_y.csv')
                
        
                X_gen_df.to_csv(output_filename_X, index=False)
                y_gen_df.to_csv(output_filename_y, index=False)
        
                print(f"Saved synthetic data for {name} in folder")
            except Exception as e:
                print(f"Error processing dataset {name}: {e}")