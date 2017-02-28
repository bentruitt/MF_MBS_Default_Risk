import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime, timedelta
from calendar import monthrange
import csv
import pdb

### Read in data and cleanup column headings
def read_data_pandas(filename, sep=','):
    df = pd.read_csv(filename, sep=sep)
    cols = df.columns.tolist()
    cols = [col.lower().replace(' ', '_') for col in cols]
    df.columns = cols
    return df

def read_data_csv(filename):
    with open(filename, 'rb') as csvfile:
        filereader = csv.reader(csvfile, dialect='excel')
        arr_list = []
        for i, row in enumerate(filereader):
             if i == 0: columns = row
             else: arr_list.append(np.array(row))
    np_data = np.array(arr_list)
    df = pd.DataFrame(data=np_data[:,:len(columns)], columns=columns)
    cols = df.columns.tolist()
    cols = [col.lower().replace(' ', '_') for col in cols]
    df.columns = cols
    return df

### Process Columns
def map_dfs(df_mflp, df_mspd, df_map):
    map_cols, mflp_cols, mspd_cols = df_map.columns.tolist()

    df_mflp = df_mflp[df_map[mflp_cols]]

    df_mspd = df_mspd[df_map[mspd_cols]]

    # Map df_mspd dlq_status_text column to corresponding values in df_MFLP
    dlq_status_text_map = {
    'Current':100,
    '90+':200,
    '< 30 ':100,
    'Grace':100,
    'Perf Balloon':100,
    '-':500, '60-89':200}
    dlq_status_text_list = df_mspd['dlq_status_text'].tolist()
    df_mspd['dlq_status_text'] = [dlq_status_text_map[x] for x in dlq_status_text_list]

    # Map df_mspd note_rate to be 0.0xxx format instead of x.xx% format
    df_mspd['note_rate'] = df_mspd['note_rate']/100.

    # Convert df_mflp quarter column to date
    eoqs = []
    this_year = datetime.today().year - 2000
    for eoq in df_mflp['quarter'].tolist():
        eoq_yr = 1900+int(int(eoq[1:3])<=this_year)*100+int(eoq[1:3])
        eoq_mo = int(eoq[-1])*3
        eoq_day = monthrange(eoq_yr, eoq_mo)[1]
        eoqs.append(str(eoq_mo) + '/' + str(eoq_day) + '/' + str(eoq_yr))
    df_mflp['quarter'] = pd.to_datetime(np.array(eoqs), errors='coerce')

    # Filter df_mspd to only newest entries
    max_qt_series = df_mflp.groupby('lnno', as_index=False)['quarter'].transform('max')
    df_mflp['max_qtr'] = max_qt_series
    df_mflp = df_mflp[df_mflp['quarter'] == df_mflp['max_qtr']]

    # pdb.set_trace()

    df_mflp.drop(['max_qtr'], inplace=True, axis=1)

    df_mflp.columns = df_map[map_cols]
    df_mspd.columns = df_map[map_cols]

    df_comb = pd.concat([df_mflp, df_mspd])

    return df_comb

### Plot histograms for data columns
def plot_histograms(df, columns, name='test', plotdir='plots/'):
    plots = len(columns)
    rem_plots = 0
    y_plots = int(np.sqrt(plots))+1
    x_plots = plots/y_plots
    rem_plots = plots - (y_plots*x_plots)
    x_plots += 1 if rem_plots else 0
    plt.figure(figsize=(y_plots*5,x_plots*5))
    plt.suptitle(name)
    for i, col in enumerate(columns):
        plt.subplot(y_plots, x_plots, i+1)
        column_list = [x for x in df[col]]
        labels, values = zip(*Counter(column_list).items())
        indexes = np.arange(len(labels))
        try:
            if type(labels[0]) == str:
                idx = np.argsort(values)
            else:
                idx = np.argsort(labels)
        except:
            pdb.set_trace()
        labels = [labels[x] for x in idx]
        values = [values[x] for x in idx]
        width = 1
        plt.bar(indexes, values, width, alpha=.5)
        plt.title(col)
        plt.xticks(indexes + width * 0.5, labels, rotation=70)
    plt.tight_layout()
    top_s = 1.-1./(3.*y_plots)
    plt.subplots_adjust(top=top_s)
    plt.savefig(plotdir + name + '_hist.png')
    plt.close()

### print table for markdown file
def print_df_md_table(df):
    headings = [('Column Name',':---'), ('Type',':---:'), ('Non-null',':---:'), ('Unique',':---:'), ('Example',':---')]
    # pdb.set_trace()
    keys = [x[0] for x in headings]
    values = [x[1] for x in headings]
    print '| ', ' | '.join(keys), ' |'
    print '| ', ' | '.join(values), ' |'
    for i, col in enumerate(df.columns.tolist()):
        row = str(df.dtypes[i:i+1]).split()[:2]
        row.append(str(df[col].notnull().sum()))
        row.append(str(df[col].nunique()))
        row.append(str(df[col].iloc[0]))
        print '| ', ' | '.join(row), ' |'

def clean_df_data(df):
    columns = df.columns.tolist()
    for col in columns:
        # set default variable default values
        blank = ''
        number, text = False, False
        # determine type of fill value used in column and replace
        fill_vals = ['-', ' ', '']
        for val in fill_vals:
            if val in df[col].unique().tolist():
                blank = val
                break
        temp_col = df[col].replace(blank, '0')
        # change column type based on column contents
        try:
            temp_col = temp_col.replace('[$,]','', regex=True)
            temp_col = temp_col.astype(float)
            if np.mod(temp_col,1).sum() == 0: # removed: temp_col.nunique() < 375 and
                temp_col = temp_col.astype(int)
            df[col] = temp_col
        except:
            temp_col = np.sort(df[col].iloc[:500])[::-1]
            try:
                # pdb.set_trace()
                temp_date = datetime.datetime.strptime(temp_col[0], '%m/%d/%Y')
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                df[col] = df[col].astype(str)
    return df

if __name__ == '__main__':
    datadir = 'data/'

    ### open data nd import into pandas dataframes
    refresh_data = str(raw_input('Would you like to refresh the data? [y/n]'))
    if refresh_data == 'y':
        mflp_file = 'mlpd_datamart_1q16.txt'
        mspd_file = 'custom_rpt_all_properties_20170222.csv'
        ## Open Multifamily Loan Performance Data
        df_mflp = read_data_pandas(datadir + mflp_file, sep='|')
        ## Open Multifamily Securitization Program Data
        df_mspd = read_data_pandas(datadir + mspd_file)
        # Clean DataFrame data
        df_mspd = clean_df_data(df_mspd)

        col_map_a = [
        ['loan_id', 'lnno','loan_id'],
        ['loan_status', 'mrtg_status', 'dlq_status_text'],
        ['published_date', 'quarter', 'distributiondate'],
        ['current_balance', 'amt_upb_endg', 'actual_balance'],
        ['original_balance', 'amt_upb_pch', 'original_note_amount'],
        ['property_state', 'code_st', 'state'],
        ['int_rate', 'rate_int', 'note_rate'],
        ['dsc_ratio', 'rate_dcr', 'dscr_(ncf)']]

        df_map = pd.DataFrame(data=col_map_a, columns=['col_a', 'mflp_col', 'mspd_col'], index=np.arange(len(col_map_a)))

        df_comb = map_dfs(df_mflp, df_mspd, df_map)
        df_comb.to_csv(datadir + 'df_comb' + '.csv')


    else:
        df_comb = pd.read_csv(datadir + 'df_comb' + '.csv')

    df_comb['label'] = df_comb['loan_status'].isin([200,300,450])
    df_comb.drop(['loan_status'], inplace=True, axis=1)

    

    ### Print table for md file
    #print_df_md_table(df_mflp)
    # print_df_md_table(df_mspd)

    #df_mdl = drop_cols(df_all)

    # plot_hists = str(raw_input('Would you like to plot a histogram for each column? [y/n]'))
    #
    # hist_columns = ['balance_range', 'dscr_range', 'dlq_status_text', 'fka_status_of_loan', 'group_id', 'loan_amortization_type', 'ltv_range', 'master_servicer', 'most_recentfinancial_indicator', 'most_recentphys_occup', 'no_time_dlq12mth', 'no_time_dlqlife', 'note_rate_range', 'occupancy_range', 'property_subtype', 'special_servicer', 'state']

    # if plot_hists == 'y':
    #     plot_histograms(df_mspd, hist_columns, name = 'Histogram of Freddie Columns')
