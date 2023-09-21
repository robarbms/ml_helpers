from data_split import split_data, sklearn_split

# Should split the data and labels into training and test sets
def test_sklear_split() -> None:
    X = [i for i in range(1, 11)]
    y = [j for j in range(11, 21)]
    X_train, X_test, y_train, y_test = sklearn_split(X, y)
    assert(len(X_train) == 8)
    assert(len(X_test) == 2)
    assert(len(y_train) == 8)
    assert(len(y_test) == 2)

