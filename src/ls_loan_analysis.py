import pandas as pd
import numpy as np

def no_space_lower(values):
    values = [value.lower().replace(" ","_") for value in values]
    return values

if __name__ == '__main__':
    data_dir = '../data/'
    data_file = '2015HMDALAR_-_National.csv'
    data_out = '2015HMDALAR_-_National_cleaned.csv'
    field_file = '2015HMDALAR_field_info.csv'
    df_fields = pd.read_csv(data_dir + field_file)
    df_fields.columns = no_space_lower(df_fields.columns.tolist())
    df_fields['field_name'] = no_space_lower(df_fields['field_name'])

    df_lar = pd.read_csv(data_dir + "2015HMDALAR_-_National.csv", header=None, names=df_fields['field_name'], dtype=object)
    nulls = ['NA',' ']
    types = {'int':int, 'float':float, 'str':str}
    for i, col in enumerate(df_lar.columns):
        temp_col = []
        new_value=None
        for value in df_lar[col]:
            try:
                new_value = map(types[df_fields['use_type']], [value])[0]
            except:
                new_value = df_fields['fill_value']
            temp_col.append(new_value)
            new_value=None
        df_lar[col] = temp_col
    df_lar.to_csv(data_dir + data_out)
    df_lar.to_pickle(data_dir + data_out) # Use pd.read_pickle(path) to get it back
