import numpy as np
def split_data(data = data, train_size= 0.70, random_state = None):

    if data is None:
        raise ValueError ("Load Data")

    if data.empty:
        raise ValueError("Data can't empty")
    
    if not 0.5<train_size<0.8:
        raise ValueError("Training size should between 0.5 - 0.8")
        
        
    # seprate input and output
    X_indices = data.iloc[:,:-1].to_numpy()
    y_indices = data.iloc[:,-1:].to_numpy()
    # print(X_indices, y_indices)
    total_sample = len(data)
    # print(total_sample)
    train_sample = int(total_sample*train_size)
    #print(train_sample)
    test_sample = int(np.ceil(total_sample - train_sample))
    # print(test_sample)

    rnd = np.random.default_rng(random_state)
    # print(rnd)
    shuffle_indices = rnd.permutation(total_sample)
    # print(shuffle_indices)
    X_train = X_indices[shuffle_indices[:train_sample]]
    y_train = y_indices[shuffle_indices[:train_sample]]

    X_test = X_indices[shuffle_indices[test_sample:]]
    y_test = y_indices[shuffle_indices[test_sample:]]

    return X_train, y_train, X_test, y_test