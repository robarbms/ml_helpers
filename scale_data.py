from sklearn.compose import make_column_selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Scales and one-hot encodes data columns
def normalize_columns(train, test, scale, one_hot):
    ct = make_column_selector(
        (MinMaxScaler(), scale),
        (OneHotEncoder(handle_unknown="ignore"), one_hot)
    )
    ct.fit(train)
    return ct.transform(train), ct.transform(test)

# Figure out the numerical columns
def getNumericalColumns(data):
    pass

# Figure out the text label columns
def getClassifiedColumns(data):
    pass

# Normalize and one-hot encode training and test data
def normalize(train, test):
    scale = getNumericalColumns(train)
    one_hot = getClassifiedColumns(train)
    return normalize_columns(train, test, scale, one_hot)
