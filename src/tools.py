import pandas as pd
import os
import re

def remove_before_double_underscore(input_string):
    result = re.sub(r"^.*?__", "", input_string)
    return result
def return_cleaned_col_names(list_of_names: list) -> list:
    cleaned_names = []
    for name in list_of_names:
        cleaned_names.append(remove_before_double_underscore(str(name)))
    return cleaned_names