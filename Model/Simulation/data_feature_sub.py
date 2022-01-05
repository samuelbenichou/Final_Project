LAST_INDEX = 0
FEATURES = []


def get_number_of_features_by_percentage(data, percentage):
    """
        Calculate the number of features of data by percentage
        @param data: the data set
        @param percentage: the percentage

        returns the number of features
    """
    data_len = data.shape[1]

    if percentage == 1:
        return data_len

    for i in range(2, data_len):
        cur_percentage = i/data_len

        if cur_percentage > percentage:
            return i-1
        elif cur_percentage == percentage:
            return i

    return 1


def update_global_feature_set(data):
    """
        Updates global features sets with current data features set
        @param: data: the data set
    """
    global FEATURES
    FEATURES.extend(data.columns.values.tolist())


def select_feature_set_from_data(data, num_of_features):
    """
        Randomly select 2 columns from data
        @param data: the data set
        @param num_of_features: num of features to take

        returns the relevant data columns
    """

    data = data[data.columns.to_series().sample(num_of_features)]
    return data


def expand_feature_set_from_data(data, num_of_features):
    """
        Expands data feature set with number of features,
        and updates the global features set.
        @param data: the data set
        @param num_of_features: num of features to take

        returns the relevant data columns
    """

    global FEATURES
    data = data.drop(FEATURES, axis=1)
    data = select_feature_set_from_data(data, num_of_features)
    update_global_feature_set(data)
    return data


def select_sub_data_from_data(data, chunk_size):
    """
        Select sub data from data
        @param data: the data
        @param chunk_size: chunk_size to take

        returns the relevant data
    """

    global LAST_INDEX
    data = data[LAST_INDEX:LAST_INDEX + chunk_size]
    LAST_INDEX += chunk_size
    return data
