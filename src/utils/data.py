import pandas as pd


def divideData(data: pd.DataFrame,
               features: list[str],
               target: str = 'RIESGO_t1',
               n_periods: int = 1):
    '''
    Divide into training/test data based on the most recent period.

    :param data: Input DataFrame
    :param features: List of feature column names
    :param target: The target variable name (default 'RIESGO_t1')
    '''
    all_periods = sorted(data["PERIODO"].unique())
    test_periods = all_periods[-n_periods:]

    train = data[~data["PERIODO"].isin(test_periods)]
    test = data[data["PERIODO"].isin(test_periods)]

    # X_train, X_test, y_train, y_test
    return (
        train[features],
        test[features],
        train[target],
        test[target]
    )
