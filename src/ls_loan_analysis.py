import pandas as pd
import numpy as np
from datetime import datetime as dt

def no_space_lower(values):
    values = [value.lower().replace(" ","_") for value in values]
    return values

if __name__ == '__main__':
    data_dir = '../data/'
    data_file = '2015HMDALAR_-_National.csv'
    data_out = '2015HMDALAR_-_National_cleaned_reduced.csv'
    data_pkl = '2015HMDALAR_-_National_cleaned_reduced.pkl'
    field_file = '2015HMDALAR_field_info.csv'
    df_fields = pd.read_csv(data_dir + field_file)
    df_fields.columns = no_space_lower(df_fields.columns.tolist())
    df_fields['field_name'] = no_space_lower(df_fields['field_name'])

    df_lar = pd.read_csv(data_dir + "2015HMDALAR_-_National.csv", header=None, names=df_fields['field_name'], dtype=object)
    nulls = ['NA',' ']
    types = {'int':int, 'float':float, 'str':str}
    for i, col in enumerate(df_lar.columns):
        print "At ", str(dt.now().strftime('%Y-%m%-%d_%H:%M')), " running column: ", i, " --- ", col
        temp_col = []
        new_value=None
        use_type=None
        for value in df_lar[col]:
            use_type = df_fields[df_fields['field_name']==col].use_type.tolist()[0]
            try:
                new_value = map(types[use_type], [value])[0]
            except:
                new_value = df_fields['fill_value']
            temp_col.append(new_value)
            new_value=None
        df_lar[col] = temp_col
    rem_cols = ['applicant_ethnicity', 'co_applicant_ethnicity', 'applicant_race_1', 'applicant_race_2', 'applicant_race_3', 'applicant_race_4', 'applicant_race_5', 'co_applicant_race_1', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4', 'co_applicant_race_5', 'applicant_sex', 'co_applicant_sex', 'hoepa_status', 'population', 'minority_population_%']
    df_lar.drop(rem_cols, axis=1, inplace=True)
    print "Saving to CSV..."
    df_lar.to_csv(data_dir + data_out)
    print "Saving to pickel..."
    df_lar.to_pickle(data_dir + data_pkl) # Use pd.read_pickle(path) to get it back
