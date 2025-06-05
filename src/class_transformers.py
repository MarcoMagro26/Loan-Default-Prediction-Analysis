from pandarallel import pandarallel
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re

pandarallel.initialize(progress_bar=True)

class GenderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        gender_mapping = {0: "Male", 1: "Female", 2: "Unknown"}
        X['Gender'] = X['Gender'].map(gender_mapping)
        return X

class ApplicationSignedHourTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['ApplicationSignedHour'] = X['ApplicationSignedHour'].apply(
            lambda x: 'Morning' if 5<= x <= 13 else
                      'Afternoon' if 14 <= x <= 19 else
                      'Evening' 
        )
        return X
    
class ApplicationSignedWeekdayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        weekday_mapping = {
            1: "Monday",
            2: "Tuesday",
            3: "Wednesday",
            4: "Thursday",
            5: "Friday",
            6: "Saturday",
            7: "Sunday"
        }
        X['ApplicationSignedWeekday'] = X['ApplicationSignedWeekday'].map(weekday_mapping)
        
        return X

class AgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['AgeGroup'] = X['Age'].apply(
            lambda x: 'Young' if x < 30 else
                      'Adult' if 30 <= x < 60 else
                      'Elderly'
        )
        return X

class AmountDifferenceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['AmountDifference'] = X['AppliedAmount'] - X['Amount']
        return X
    
class EducationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        education_mapping = {
            -1: 'Unknown',
            0: 'Unknown',
            1: 'Primary',
            2: 'Basic',
            3: 'Vocational',
            4: 'Secondary',
            5: 'Higher'
        }
        X = X.copy()
        X['Education'] = X['Education'].map(education_mapping)
        return X
    
    
class BidsAnalysisTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['BidsTotal'] = X['BidsPortfolioManager'] + X['BidsApi'] + X['BidsManual']
        return X
    
class LanguageCountryMatchTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Dizionario di mapping per LanguageCode
        self.language_mapping = {
            1: "Et", 2: "En", 3: "Ru", 4: "Fi", 5: "De", 6: "Es", 7: "Pl", 8: "Lv", 9: "Sk",
            10: "Sl", 11: "Bg", 12: "Hr", 13: "Cs", 14: "Da", 15: "Fr", 16: "El", 17: "Hu",
            18: "Lt", 19: "Nl", 20: "Pt", 21: "Ro", 22: "Sv", 23: "It", 24: "No", 25: "Zu", 26: "Ja"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["LanguageCode"] = X["LanguageCode"].map(self.language_mapping)
        X["LanguageCountryMatch"] = X.apply(self.check_language_country, axis=1)
        return X

    @staticmethod
    def check_language_country(row):
        if row["Country"] == "EE" and row["LanguageCode"] == "Et":
            return "Citizen"
        elif row["Country"] == "FI" and (row["LanguageCode"] == "Fi" or row["LanguageCode"] == "Sv"):
            return "Citizen"
        elif row["Country"] == "ES" and row["LanguageCode"] == "Es":
            return "Unknown"
        elif row["Country"] == "SK" and row["LanguageCode"] == "Sk":
            return "Citizen"
        elif row["Country"] == "NL" and row["LanguageCode"] == "Nl":
            return "Citizen"
        else:
            return "Foreign"

class NewCreditCustomerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["NewCreditCustomer"] = X["NewCreditCustomer"].map({1: 0, 0: 1})
        return X


class VerificationTypeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        verification_mapping = {
            0:"Unknown",
            1:"NotVerified",
            2:"VerifiedByPhone",
            3:"VerifiedByOtherDocument",    
            4:"VerifiedByBankStatement",
        }
        X = X.copy()
        X["VerificationType"] = X["VerificationType"].map(verification_mapping)
        return X

class OccupationAreaTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        occupation_mapping = {
            -1: "Unknown",
            0: "Unknown",
            1: "Other",
            2: "PrimarySector",  # Mining
            3: "SecondarySector",  # Processing
            4: "SecondarySector",  # Energy
            5: "SecondarySector",  # Utilities
            6: "SecondarySector",  # Construction
            7: "TertiarySector",  # Retail
            8: "TertiarySector",  # Transport
            9: "TertiarySector",  # Hospitality
            10: "TertiarySector",  # Telecom
            11: "TertiarySector",  # Finance
            12: "TertiarySector",  # RealEstate
            13: "TertiarySector",  # Research
            14: "TertiarySector",  # Administrative
            15: "TertiarySector",  # CivilService
            16: "TertiarySector",  # Education
            17: "TertiarySector",  # Healthcare
            18: "TertiarySector",  # Art
            19: "PrimarySector",  # Agriculture
        }
        X = X.copy()
        X["OccupationArea"] = X["OccupationArea"].map(occupation_mapping)
        return X

class RatingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rating_mapping = {
            "AA": "LowRisk",
            "A": "LowRisk",
            "B": "LowRisk",
            "C": "MediumRisk",
            "D": "MediumRisk",
            "E": "HighRisk",
            "F": "HighRisk",
            "HR": "Unknown" 
        }
        X = X.copy()
        X["Rating"] = X["Rating"].map(rating_mapping)
        return X

class HomeOwnershipTypeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        home_ownership_mapping = {
            -1: "Unknown",
            0: "Homeless",
            1: "Owner",
            7: "Owner",
            8: "Owner",
            9: "Owner",
            3: "Tenant",
            4: "Tenant",
            5: "Tenant",
            6: "Tenant",
            2: "Other",
            10: "Other"
        }
        X = X.copy()
        X["HomeOwnershipType"] = X["HomeOwnershipType"].map(home_ownership_mapping)
        return X
    

    
class CreditScoreEsMicroLTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        credit_score_mapping = {
            "M": "Unknown",
            "M1": "HighScore", #LowRisk
            "M2": "HighScore",
            "M3": "HighScore",
            "M4": "MediumScore", #MediumRisk
            "M5": "MediumScore",
            "M6": "MediumScore",
            "M7": "MediumScore",
            "M8": "LowScore",   #HighRisk
            "M9": "LowScore",
            "M10": "LowScore"
        }
        X = X.copy()
        X["CreditScoreEsMicroL"] = X["CreditScoreEsMicroL"].map(credit_score_mapping)
        return X

class LoanDurationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Categorizza la variabile LoanDuration in poche classi.
        
        Classi:
        - "Short Term" (<= 12 mesi)
        - "Medium Term" (13-36 mesi)
        - "Long Term" (37-60 mesi)
        - "Very Long Term" (> 60 mesi)
        """
        X = X.copy()
        bins = [0, 12, 36, 60, float('inf')]  # Intervalli
        labels = ["Short Term", "Medium Term", "Long Term", "Very Long Term"]  # Etichette
        X["LoanDuration_Category"] = pd.cut(X["LoanDuration"], bins=bins, labels=labels, right=True)
        X["LoanDuration_Category"] = X["LoanDuration_Category"].astype(str)  # Converti in stringa
        return X
    
class ColumnDropperTransformer:
    def __init__(self, columns: list):
        self.columns: list = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self
