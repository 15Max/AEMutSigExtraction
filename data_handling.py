import pandas as pd
import numpy as np

def data_augmentation(X : pd.DataFrame , augmentation : int =5) -> pd.DataFrame:
    '''
    A function to perform data augmentation on the input data. Here X is expected to be a pandas DataFrame that contains
    the count data, rows represent signatures and columns patients. The data augmentation is performed by bootstrapping
    the data and then concatenating the bootstrapped data to the original data, obtaining a "stacked" matrix

    Parameters:
    X (pd.DataFrame): The input data to be augmented
    augmentation (int): The number of times to augment the data
    returns (pd.DataFrame): The augmented data
    '''

    X_augmented=[]
    
    for i in range(augmentation):
        X_bootstrapped=[]
        for col in X.columns:
            N = int(round(np.sum(X[col])))
            p = X[col] / N 
            X_bootstrapped.append(np.random.multinomial(N, p))

        X_bootstrapped = np.transpose(np.array(X_bootstrapped))

        # Make sure that the columns have original names: original name + _augmented + i

        X_bootstrapped = pd.DataFrame(X_bootstrapped, columns=[str(col) + '_aug_' + str(i) for col in X.columns])
        X_augmented.append(pd.DataFrame(X_bootstrapped, dtype= 'int64'))

    X_aug = pd.concat(X_augmented, axis=1)

    return X_aug



def data_normalization(X : pd.DataFrame) -> pd.DataFrame:
    '''
    A function that normalizes the input data. Here X is expected to be a pandas DataFrame that contains
    the count data, rows represent signatures and columns patients.

    Parameters:
    X (pd.DataFrame): The input data to be normalized
    returns (pd.DataFrame): The normalized data
    '''

    # Calculate the total number of mutations per patient

    total_mutations = X.sum(axis=1)

    # Repeat the total number of mutations for each patient

    total_mutations = pd.concat([total_mutations] * X.shape[1], axis=1)

    # Set the column names of the total_mutations DataFrame to match the input data

    total_mutations.columns = X.columns

    # Normalize the data using the log-ratio transformation (Count data follows a Poisson distribution, in theory, so
    # the log-ratio transformation feels appropriate)

    norm_data = X / total_mutations * np.log2(total_mutations)

    return np.array(norm_data, dtype='float64')


if __name__ == "__main__":

    # Randomly generate a dataset 10 rows, 10 columns of count data (non-negative integers)

    data = np.random.randint(1, 100, size=(96, 2))
    data = pd.DataFrame(data)
    
    print("Original Data:")

    print(data)

    # Perform data augmentation on the dataset

    augmented_data = data_augmentation(data, 1)

    # Display the original and augmented datasets

    print("\nAugmented Data:")

    print(augmented_data)

    print("\nNormalized Data:")

    # Normalize the data using the log-ratio transformation

    data = pd.DataFrame(data)

    # data = data.transpose()

    normalized_data = data_normalization(data)

    print(normalized_data)

    print("\nNormalized Data Shape:")

    print(normalized_data.shape)