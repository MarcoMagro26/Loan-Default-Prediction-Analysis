import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def __transform_to_datetime(df: pd.DataFrame, column_names: list) -> pd.DataFrame:
    """
    Converts specified columns to datetime format.
    :param df: Input DataFrame.
    :param column_names: List of column names to transform.
    :return: DataFrame with specified columns converted to datetime.
    """
    df = df.copy()
    for column in column_names:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors='coerce')
        else:
            raise ValueError(f"The column '{column}' does not exist in the DataFrame.")
    return df
fun_tr_transform_to_datetime = FunctionTransformer(__transform_to_datetime, kw_args={"column_names": ["LoanDate", "FirstPaymentDate", "MaturityDate_Original", "MaturityDate_Last", "LastPaymentOn"]})

def __transform_to_int(df: pd.DataFrame, column_names: list) -> pd.DataFrame:
    """
    Converts specified columns to integer format.
    :param df: Input DataFrame.
    :param column_names: List of column names to transform.
    :return: DataFrame with specified columns converted to integers.
    """
    df = df.copy()
    for column in column_names:
        if column in df.columns:
            df[column] = df[column].astype(int)
    return df

fun_tr_transform_to_int = FunctionTransformer(__transform_to_int, kw_args={"column_names": ["VerificationType", "Gender", "Education", "OccupationArea", "HomeOwnershipType", "PrincipalOverdueBySchedule", "ModelVersion", "NoOfPreviousLoansBeforeLoan", "AmountOfPreviousLoansBeforeLoan", "PreviousEarlyRepaymentsCountBeforeLoan", "BidsPortfolioManager", "BidsApi", "BidsManual", "AppliedAmount", "Amount", "IncomeFromPrincipalEmployer", "IncomeFromPension", "IncomeFromFamilyAllowance", "IncomeFromSocialWelfare", "IncomeFromLeavePay", "IncomeFromChildSupport", "IncomeOther", "IncomeTotal", "NoOfPreviousLoansBeforeLoan","NewCreditCustomer", "ActiveScheduleFirstPaymentReached", "Restructured"]})


def __round_to_two_decimals(df: pd.DataFrame, column_names: list) -> pd.DataFrame:
    """
    Rounds specified columns to two decimal places.
    :param df: Input DataFrame.
    :param column_names: List of column names to transform.
    :return: DataFrame with specified columns rounded to two decimals.
    """
    df = df.copy()
    for column in column_names:
        if column in df.columns:
            df[column] = df[column].round(2)
    return df

fun_tr_round_to_two_decimals = FunctionTransformer(__round_to_two_decimals, kw_args={"column_names": ["LiabilitiesTotal","Interest", "DebtToIncome", "FreeCash", "PlannedInterestTillDate", "ExpectedLoss", "LossGivenDefault", "ExpectedReturn", "ProbabilityOfDefault"]})


# Dropping features
def __drop_features_with_many_nan(x: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, returns the dataframe without the features with at least
    half of the rows Nan
    :param x: dataframe
    :return: dataframe without the partially-Nan columns
    """
    df = x.copy()
    nulls_summary = pd.DataFrame(df.isnull().sum())
    more_than_null_features = nulls_summary.loc[
        nulls_summary.iloc[:, 0] > df.shape[0] * 0.1, :
    ].index.tolist()
    print("\nDropping features:")
    print(more_than_null_features)
    return x.drop(more_than_null_features, axis=1)

fun_tr_drop_features_with_many_nan = FunctionTransformer(__drop_features_with_many_nan)

