from sklearn.model_selection import train_test_split

# Function for splitting data
def split_data(data):
    ten_percent = len(data) // 10
    train = data[:ten_percent * 8]
    valid = data[ten_percent * 8: ten_percent * 9]
    test = data[ten_percent * 9:]

    return train, valid, test

# Splitting with sklearn
def sklearn_split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
