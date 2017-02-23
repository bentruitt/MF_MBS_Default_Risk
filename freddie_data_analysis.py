import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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
def drop_cols(df, columns):
    # drop_cols = ['ad_group_name', 'match_type', 'currency', 'same_sku_units_ordered_within_1-week_of_click', 'other_sku_units_ordered_within_1-week_of_click', 'same_sku_units_product_sales_within_1-week_of_click', 'other_sku_units_product_sales_within_1-week_of_click']
    # df.drop(labels=drop_cols, axis=1, inplace=True)
    # df.rename(columns={"orders_placed_within_1-week_of_a_click":"orders", "product_sales_within_1-week_of_a_click":"sales", "conversion_rate_within_1-week_of_a_click":"conv_rate"}, inplace=True)

    pass

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
        # new_labels = []
        # for label in labels:
        #     if type(label) == str:
        #         new_label = [x[0] for x in label.split()]
        #         new_labels.append(label)
        # labels = new_labels
        width = 1
        plt.bar(indexes, values, width, alpha=.5)
        plt.title(col)
        plt.xticks(indexes + width * 0.5, labels, rotation=70)
    plt.tight_layout()
    top_s = 1.-1./(3.*y_plots)
    plt.subplots_adjust(top=top_s)
    plt.savefig(plotdir + name + '_hist.png')
    plt.close()

# def plot_histograms(df, columns = [], name = 'test', plotdir='plots/'):
#     plt.figure()
#     df.plot.hist(alpha=0.5)
#     if name == None:
#         if len(columns)>1:
#             name = [x[0] for x in columns].join()
#         else:
#             name = columns[0]
#     plt.savefig(plotdir + name + '_hist.png')

if __name__ == '__main__':
    datadir = 'data/'
    mlpd_file = 'mlpd_datamart_1q16.txt'

    open_data = str(raw_input('Would you like to open ' + mlpd_file + '? [y/n]'))
    if open_data == 'y':
        df_pd = read_data_pandas(datadir + mlpd_file, sep='|')

    open_data = str(raw_input('Would you like to open the investor data? [y/n]'))
    if open_data == 'y':
        all_loans_file = 'custom_rpt_all_properties_20170222.csv'
        #df_all = read_data_csv(datadir + all_loans_file)
        df_all = read_data_pandas(datadir + all_loans_file)

    ## Print table for md file
    for i, col in enumerate(df_all.columns.tolist()):
        # pdb.set_trace()
        row = str(df_all.dtypes[i:i+1]).split()[:2]
        row.append(str(df_all[col].notnull().sum()))
        row.append(str(df_all[col].nunique()))
        print '| ', ' | '.join(row), ' |'


    # plot_hists = str(raw_input('Would you like to plot a histogram for each column? [y/n]'))
    # all_columns = df.columns.tolist()
    # remove_columns = ['code_sr', ]
    # if plot_hists == 'y':
    #     plot_histograms(df, plot_columns, name = 'Histogram of Freddie Columns')
    # for col in all_columns:
    #     print "Column '", col, "' has ", df[col].nunique(), " values"
    #
    # plot_histograms(df, name='all_columns')
