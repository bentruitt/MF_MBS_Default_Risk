import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import datetime
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
def drop_cols(df):
    drop_columns = ['#_properties', '2nd_preceding_fydscr_(ncf)', 'balloon', 'city', 'current%_of_deal', 'current_bal_rank', 'current_endingsched_balance', 'currentloan_balance', 'cut-off_dateloan_balance', 'distributiondate', 'dlq._status', 'int._only', 'maturity_date', "most_recent_financial_'as_of'_end_date_", "most_recent_financial_'as_of'_start_date_", 'occupancy_date', 'occupancy_source', 'operatingtrust_advisor', 'original_occupancy_date', 'paid_thru_date', 'payment_freq', "preceding_fiscalyear_financial_'as_of'_date_", 'preceding_fy_ncf', 'preceding_fiscalyear_noi', 'preceding_fy_dscr_(ncf)', 'property_name', 'prospectus_id', 'remaining_term', 'seasoning', 'seasoning_range', "second_preceding_fiscal_year_financial_'as_of'_date_", 'second_preceding_fy_ncf_(dscr)', 'second_preceding_fy_ncf', 'second_precedingfiscal_year_noi', 'servicer_watchlistcode', 'total_reserve_bal', 'zip_code', 'property_address', 'property_city', 'revenue_at_contribution', 'operating_expenses_at_contribution', 'noi_at_contribution', 'dscr_(noi)_at_contribution', 'ncf_at_contribution', 'dscr_(ncf)_at_contribution', 'second_preceding_fy_revenue', 'second_preceding_fy_operating_exp', 'second_preceding_fy_debt_serv_amt', 'preceding_fy_revenue', 'preceding_fy_operating_exp', 'preceding_fy_debt_svc_amount', 'most_recent_revenue', 'most_recent_operating_expenses', 'most_recent_debt_service_amount']

    df.drop(labels=drop_columns, axis=1, inplace=True)

    return df

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
    open_data = str(raw_input('Would you like to open the data? [y/n]'))
    if open_data == 'y':
        mflp_file = 'mlpd_datamart_1q16.txt'
        mspd_file = 'custom_rpt_all_properties_20170222.csv'
        ## Open Multifamily Loan Performance Data
        df_mflp = read_data_pandas(datadir + mflp_file, sep='|')
        # Open Multifamily Securitization Program Data
        df_mspd = read_data_pandas(datadir + mspd_file)

    ### Print table for md file
    #print_df_md_table(df_mflp)
    # print_df_md_table(df_mspd)

    ### Clean DataFrame data
    df_mspd = clean_df_data(df_mspd)

    #df_mdl = drop_cols(df_all)

    plot_hists = str(raw_input('Would you like to plot a histogram for each column? [y/n]'))

    hist_columns = ['balance_range', 'dscr_range', 'dlq_status_text', 'fka_status_of_loan', 'group_id', 'loan_amortization_type', 'ltv_range', 'master_servicer', 'most_recentfinancial_indicator', 'most_recentphys_occup', 'no_time_dlq12mth', 'no_time_dlqlife', 'note_rate_range', 'occupancy_range', 'property_subtype', 'special_servicer', 'state']

    if plot_hists == 'y':
        plot_histograms(df_mspd, hist_columns, name = 'Histogram of Freddie Columns')
