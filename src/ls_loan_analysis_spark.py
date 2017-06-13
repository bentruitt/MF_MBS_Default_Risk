from multiprocessing import Pool, cpu_count
import pyspark as ps
import pandas as pd
import numpy as np
from datetime import datetime as dt
from pdb import set_trace

def no_space_lower(values):
    values = [value.lower().replace(" ","_") for value in values]
    return values

def casting_function(field, use_type, fill_value):
    type_dict = {'int':int, 'float':float, 'str':str}
    try:
        return map(type_dict[use_type], [field])[0]
    except:
        return fill_value

if __name__ == '__main__':

    ### Set location variables
    data_dir = '~/GitData/MF_MBS_Default_Risk/'

    ### Set file name variables
    data_file = '2015HMDALAR_-_National.csv'
    data_out = '2015HMDALAR_-_National_cleaned_reduced.csv'
    data_pkl = '2015HMDALAR_-_National_cleaned_reduced.pkl'
    field_file = '2015HMDALAR_field_info.csv'

    ### Set static variables
    rem_cols = ['applicant_ethnicity', 'co_applicant_ethnicity', 'applicant_race_1', 'applicant_race_2', 'applicant_race_3', 'applicant_race_4', 'applicant_race_5', 'co_applicant_race_1', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4', 'co_applicant_race_5', 'applicant_sex', 'co_applicant_sex', 'hoepa_status', 'population', 'minority_population_%']

    ### Open and get df_fields table ready for use
    df_fields = pd.read_csv(data_dir + field_file)
    df_fields.columns = no_space_lower(df_fields.columns.tolist())
    df_fields['field_name'] = no_space_lower(df_fields['field_name'])
    df_fields['drop_column'] = df_fields['field_name'].isin(rem_cols)

    ### initialize Spark session
    spark = ps.sql.SparkSession.builder \
                                .master("local[6]") \
                                .appName("len_stand") \
                                .getOrCreate()
    sc = spark.sparkContext

    ### Create RDD of '2015HMDALAR_-_National.csv' and classify columns as correct datatypes and identify null values

    rdd_classed = sc.textFile(data_dir + data_file) \
                    .map(lambda rowstr: rowstr.split(",")) \
                    .map(lambda row: map(casting_function, row, df_fields['use_type'], df_fields['fill_value']))

    df_lar = rdd_classed.toDF(df_fields['field_name'].tolist())
    df_lar = df_lar.select([col for col in df_lar.columns if col not in rem_cols])

    ### Test run
    # for row in sc.textFile(data_dir + data_file).take(5):
    #     row = row.split(",")
    #     new_row = map(casting_function, row, df_fields['use_type'], df_fields['fill_value'])
    #     print new_row

    ### Convert columns to correct types and identify null values
    # df_lar = pd.read_csv(data_dir + "2015HMDALAR_-_National.csv", header=None, names=df_fields['field_name'], dtype=object)


    # df_lar.drop(rem_cols, axis=1, inplace=True)
    # print "Saving to CSV..."
    # df_lar.to_csv(data_dir + data_out)
    # print "Saving to pickel..."
    # df_lar.to_pickle(data_dir + data_pkl) # Use pd.read_pickle(path) to get it back
