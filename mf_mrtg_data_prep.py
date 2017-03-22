import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime, timedelta
from calendar import monthrange
import csv
from pdb import set_trace

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

    prob_loan_ids = df_mspd['loan_id'][(df_mspd['no_time_dlqlife']>0) | (df_mspd['date_added_to_servicer_watchlist_']!="-")].tolist()

    map_cols, mflp_cols, mspd_cols = df_map.columns.tolist()

    df_mflp = df_mflp[df_map[mflp_cols]]

    df_mspd = df_mspd[df_map[mspd_cols]]

    # Map df_mspd dlq_status_text column to corresponding values in df_MFLP
    # 100 = Current or less than 60 day delinquent
    # 200 = 60 or more days delinquent
    # 300 = Foreclosure
    # 350 = Problem Loans
    # 450 = Real estate owned
    # 500 = Closed
    dlq_status_text_map = {
    'Current':100,
    '90+':200,
    '< 30 ':350,
    'Grace':100,
    'Perf Balloon':350,
    '-':100, '60-89':200}

    dlq_status_text_list = df_mspd['dlq_status_text'].tolist()
    df_mspd['dlq_status_text'] = [dlq_status_text_map[x] for x in dlq_status_text_list]

    # Map df_mspd note_rate to be 0.0xxx format instead of x.xx% format
    df_mspd['note_rate'] = np.divide(df_mspd['note_rate'], 100.)

    # Convert df_mflp quarter column to date
    eoqs = []
    this_year = datetime.today().year - 2000
    for eoq in df_mflp['quarter'].tolist():
        eoq_yr = 1900+int(int(eoq[1:3])<=this_year)*100+int(eoq[1:3])
        eoq_mo = int(eoq[-1])*3
        eoq_day = monthrange(eoq_yr, eoq_mo)[1]
        eoqs.append(str(eoq_mo) + '/' + str(eoq_day) + '/' + str(eoq_yr))
    df_mflp['quarter'] = pd.to_datetime(np.array(eoqs), errors='coerce')
    # set_trace()
    # Get dataframe of delinquency status
    df_mflp_dlq = df_mflp.loc[:,['lnno', 'mrtg_status']]

    # Filter df_mflp to only newest entries
    max_qt_series = df_mflp.groupby('lnno', as_index=False)['quarter'].transform('max')
    df_mflp['max_qtr'] = max_qt_series
    df_mflp = df_mflp[df_mflp['quarter'] == df_mflp['max_qtr']]

    # Assign mrtg status in df_mflp to highest level prior to 500
    new_mrtg_status = []
    for ln_id in df_mflp['lnno'].tolist():
        dlq_status_arr = df_mflp_dlq[df_mflp_dlq['lnno'] == ln_id]['mrtg_status'].unique()
        if len(dlq_status_arr) > 1:
            dlq_status_arr = np.delete(dlq_status_arr, np.where(dlq_status_arr==500))
        new_mrtg_status.append(dlq_status_arr.max())
    df_mflp['mrtg_status'] = np.array(new_mrtg_status)

    df_mflp.drop(['max_qtr'], inplace=True, axis=1)

    # add engineered columns to df_mflp
    df_mflp.columns = df_map[map_cols]
    df_mflp['published_date'] = pd.to_datetime(df_mflp['published_date'], errors='coerce')
    df_mflp['freddie_held'] = 1.0
    df_mflp['principal_paydown'] = df_mflp['original_balance'] - df_mflp['current_balance']
    # define label to be loans beyond 60 days late and any that have defaulted in the past
    df_mflp['label'] = [float(x) for x in df_mflp['loan_status'].isin([200,300,350,450])]
    df_mflp.drop(['loan_status'], inplace=True, axis=1)
    df_mflp.to_csv(datadir + 'df_mflp_labeled' + '.csv')

    # add engineered columns to df_mspd
    df_mspd.columns = df_map[map_cols]
    df_mspd['published_date'] = pd.to_datetime(df_mspd['published_date'], errors='coerce')
    df_mspd['freddie_held'] = 0.0
    df_mspd['principal_paydown'] = df_mspd['original_balance'] - df_mspd['current_balance']
    # define label to be loans beyond 60 days late and any that have defaulted in the past
    df_mspd['label'] = [float(x) for x in ((df_mspd['loan_status'].isin([200,300,350,450])) | (df_mspd['loan_id'].isin(prob_loan_ids)))]
    df_mspd.drop(['loan_status'], inplace=True, axis=1)
    df_mspd.to_csv(datadir + 'df_mspd_labeled' + '.csv')

    df_comb = pd.concat([df_mflp, df_mspd])
    df_comb.drop(['current_balance', 'principal_paydown'], inplace=True, axis=1)
    df_comb.set_index(np.arange(df_comb.shape[0]), drop=True, inplace=True)
    df_comb.to_csv(datadir + 'df_comb_labeled' + '.csv')

    return df_mflp, df_mspd, df_comb

### Plot histograms for data columns
def plot_histograms(df, columns, name='test', plotdir='plots/'):
    filename = name.replace(" ","_")
    rem_columns = []
    for col in columns:
        if (df[col].nunique() > 60 and type(df[col][0])==str) or (df[col].nunique()==len(df[col]) or df[col].nunique() < 3):
            rem_columns.append(col)
    columns = [x for x in columns if x not in rem_columns]
    plots = len(columns)
    rem_plots = 0
    y_plots = int(np.sqrt(plots))+1
    x_plots = plots/y_plots
    rem_plots = plots - (y_plots*x_plots)
    x_plots += 1 if rem_plots else 0
    plt.figure(figsize=(y_plots*5,x_plots*5))
    plt.suptitle(name)
    for i, col in enumerate(columns):
        column_list = [x for x in df[col]]
        plt.subplot(y_plots, x_plots, i+1)
        if df[col].nunique() > 60:
            try:
                plt.hist(df[col],
                    range=(df[col].min(), df[col].max()),
                    bins=50,
                    alpha=0.75)
            except:
                set_trace()
        else:
            labels, values = zip(*Counter(column_list).items())
            indexes = np.arange(len(labels))
            if type(labels[0]) == str:
                idx = np.argsort(values)
            else:
                idx = np.argsort(labels)
            labels = [labels[x] for x in idx]
            values = [values[x] for x in idx]
            width = 1
            plt.bar(indexes, values, width, alpha=.5)
            plt.xticks(indexes + width * 0.5, labels, rotation=70)
        plt.title(col)
    plt.tight_layout()
    top_s = 1.-1./(3.*y_plots)
    plt.subplots_adjust(top=top_s)
    plt.savefig(plotdir + filename + '_hist.png')
    plt.close()

### print table for markdown file
def print_df_md_table(df):
    headings = [('Column Name',':---'), ('Type',':---:'), ('Non-null',':---:'), ('Unique',':---:'), ('Example',':---')]
    # set_trace()
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

def clean_mspd_data(df):
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
                # set_trace()
                temp_date = datetime.datetime.strptime(temp_col[0], '%m/%d/%Y')
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                df[col] = df[col].astype(str)
    ltv_col = df['uw_ltv'].replace('[%]','', regex=True)
    ltv_col = np.divide(ltv_col.astype(float),100.)
    df['uw_ltv'] = ltv_col
    special_svcr = [x.lower().replace(' ', '_') for x in df['special_servicer']]
    df['special_servicer'] = np.array(special_svcr)
    loan_ct = Counter(df['loan_id'])
    dup_ids = []
    for key, value in loan_ct.iteritems():
        if value>1:
            dup_ids.append(key)
    for dup_id in dup_ids:
        df_temp = df[df['loan_id']==dup_id]
        for uzip in df_temp['zip_code'].tolist():
            state = df_temp[df_temp['zip_code']==uzip]['state'].tolist()[0].upper()
            city = df_temp[df_temp['zip_code']==uzip]['property_city'].tolist()[0].upper()
            zips = df_zip[(df_zip['state']==state) & (df_zip['city']==city)]['zipcode'].tolist()
            if int(uzip) not in zips:
                df.drop(df_temp[df_temp['zip_code']==uzip].index, inplace=True)
    return df

if __name__ == '__main__':
    datadir = 'data/'

    ### determine whether or not to plot histograms of MSPD
    plot_hists = bool(raw_input('Would you like to plot a histogram for each MSPD column? [y/n]')=='y')

    ### open data and import into pandas dataframes
    mflp_file = 'mlpd_datamart_1q16.txt'
    mspd_file = 'custom_rpt_all_properties_20170222.csv'
    zip_file = 'zipcode-database.csv'
    ## open zipcode data
    df_zip = read_data_pandas(datadir + zip_file)
    df_zip = df_zip[['zipcode', 'city', 'state']]
    ## Open Multifamily Loan Performance Data
    df_mflp = read_data_pandas(datadir + mflp_file, sep='|')
    df_mflp['special_svcr'] = 'freddie_mac'
    # Clean DataFrame data
    # df_mflp = clean_df_data(df_mflp)
    ## Open Multifamily Securitization Program Data
    df_mspd = read_data_pandas(datadir + mspd_file)
    # Clean DataFrame data
    df_mspd = clean_mspd_data(df_mspd)
    df_mspd_o = df_mspd.copy()
    df_mspd_o.to_csv(datadir + 'df_mspd_o.csv')

    ### map input tables to new output table for analysis
    # New table | MFLP table | MSPD table
    col_map_a = [
    ['loan_id', 'lnno','loan_id'],
    ['loan_status', 'mrtg_status', 'dlq_status_text'],
    ['published_date', 'quarter', 'distributiondate'],
    ['current_balance', 'amt_upb_endg', 'actual_balance'],
    ['original_balance', 'amt_upb_pch', 'original_note_amount'],
    ['property_state', 'code_st', 'state'],
    ['orig_int_rate', 'rate_int', 'note_rate'],
    ['orig_dscr', 'rate_dcr', 'uw_dscr_(ncf)amortizing'],
    ['orig_ltv', 'rate_ltv', 'uw_ltv'],
    ['special_servicer', 'special_svcr', 'special_servicer']]

    df_map = pd.DataFrame(data=col_map_a, columns=['col_a', 'mflp_col', 'mspd_col'], index=np.arange(len(col_map_a)))

    df_mflp, df_mspd, df_comb = map_dfs(df_mflp, df_mspd, df_map)

    # add additional columns to df_mspd (securitized set)
    df_mspd['orig_occ_rate'] = np.divide(df_mspd_o['original_occupancy__rate'], 100.)
    df_mspd['most_rct_occ_rate'] = np.divide(df_mspd_o['most_recentphys_occup'], 100.)
    df_mspd['occ_rate_delta'] = df_mspd['most_rct_occ_rate'] - df_mspd['orig_occ_rate']

    df_mspd['orig_noi'] = df_mspd_o['noi_at_contribution']
    df_mspd['most_rct_noi'] = df_mspd_o['most_recent_noi']
    df_mspd['noi_delta'] = df_mspd['most_rct_noi'] - df_mspd['orig_noi']

    df_mspd['most_rct_value'] = df_mspd_o['most_recent_value']
    df_mspd['orig_value'] = df_mspd['original_balance'] / df_mspd['orig_ltv']
    df_mspd['orig_value'][df_mspd['orig_ltv']==0] = df_mspd['most_rct_value']
    df_mspd['value_delta'] = df_mspd['most_rct_value'] - df_mspd['orig_value']

    df_mspd['most_rct_ltv'] = df_mspd['current_balance'] / df_mspd['most_rct_value']
    df_mspd['ltv_delta'] = df_mspd['most_rct_ltv'] - df_mspd['orig_ltv']

    df_mspd['most_rct_debt_serv'] = df_mspd_o['most_recent_debt_service_amount']
    df_mspd['orig_debt_serv'] = df_mspd['orig_noi'] / df_mspd['orig_dscr']
    df_mspd['debt_serv_delta'] = df_mspd['most_rct_debt_serv'] - df_mspd['orig_debt_serv']

    df_mspd['most_rct_dscr'] = df_mspd_o['most_recentdscr_(ncf)']
    df_mspd['dscr_delta'] = df_mspd['most_rct_dscr'] - df_mspd['orig_dscr']

    df_mspd = df_mspd.replace([-np.inf, np.inf],0)

    df_mspd.to_csv(datadir + 'df_mspd_labeled_built_up.csv')

    if plot_hists:
        hist_columns = df_mspd.columns.tolist()
        plot_histograms(df_mspd, hist_columns, name = 'Histograms of MSPD Columns')
