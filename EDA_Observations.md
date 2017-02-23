# Observations of Data During EDA

## Multifamily Loan Performance Dataset (MFLP) - mlpd_datamart_1q16.txt
**Data Source: ** [Investor Tools - Loan Performance Database](http://www.freddiemac.com/multifamily/investors/reporting.html)

The Database provides historical information on a subset of the Freddie Mac Multifamily whole loan portfolio since 1994.  It includes information on original loan terms; identifiers for prepaid loans, defaulted loans and delinquencies; property information; and dates of real estate owned (REO) sales.

There are 338,445 rows of data covering 11,570 unique loans. Multiple loans may be related to the same property.

In [21]: df['lnno'].nunique()  
Out[21]: 11570

**Column Data Types:**

| **Column Name**      | **Non-Null** | **Type**     | **Convert** |
| -------------------- |:------------:|:------------:|:-----------:|
| lnno                 | 338445       |  int64       | leave       |
| quarter              | 338445       |  object(str) | datetime    |
| mrtg_status          | 338445       |  int64       | get_dummies |
| amt_upb_endg         | 328999       |  float64     |  leave      |
| liq_dte              | 9160         |  object(str) | datetime    |
| liq_upb_amt          | 9164         |  float64     | leave       |
| dt_sold              | 73           |  object(str) | datetime    |
| cd_fxfltr            | 130063       |  object(str) | get_dummies |
| cnt_amtn_per         | 338428       |  float64     | leave       |
| cnt_blln_term        | 338428       |  float64     | leave       |
| cnt_io_per           | 338419       |  float64     | leave       |
| dt_io_end            | 98108        |  object(str) | datetime    |
| cnt_mrtg_term        | 338445       |  int64       | leave       |
| cnt_rsdntl_unit      | 338411       |  float64     | leave       |
| cnt_yld_maint        | 338445       |  int64       | leave       |
| code_int             | 338445       |  object      | get_dummies |
| rate_dcr             | 338290       |  float64     | leave       |
| rate_int             | 338416       |  float64     | leave       |  
| rate_ltv             | 338392       |  float64     | leave       |
| amt_upb_pch          | 338445       |  float64     | leave       |
| dt_fund              | 338445       |  object(str) | datetime    |
| dt_mty               | 338445       |  object(str) | datetime    |
| code_st              | 38445        |  object(str) | leave       |
| geographical_region  | 338048       |  object(str) | leave       |
| **id_link_grp**      | 66308        |  float64     | leave       |
| code_sr              | 19684        |  object(str) | get_dummies |
| reo_operating_expinc | 74           |  float64     | leave       |
| prefcl_fcl_expinc    | 74           |  float64     | leave       |
| selling_expinc       | 74           |  float64     | leave       |
| sales_price          | 73           |  float64     | leave       |


**'id_link_grp'** links together loans that are related to the same property

**Pros / Cons**
Pros - This dataset is well compiled and would require little data cleaning and adjustments before running through models.

Cons - This is a subset of the overall Freddie Mac multifamily loans issued and may not be representative of the typical loan funded by Freddie Mac and sold to investors. These are only the loans of which Freddie Mac has retained ownership. There could be a variety of reasons that they would have retained ownership of these loans. These loans could be for types of properties that are not easily bundled with other properties for securitization. This could apply to senior housing properties with some level of care (i.e. assisted living, nursing homes, memory care, etc.). This could also apply to affordable housing projects with complex capital structures and mechanisms that would potentially deter the secondary capital markets from investing in their securities. A few examples of some characteristics that could deter capital markets would be rent restrictions, phasing off tax abatements, land leases, or rent subsidies from entities with low credit ratings, such as, a bankrupt county or city. I would Freddie Mac not cherry pick these loans to inflate their performance data. The federal government did take control of both Freddie Mac and Fannie Mae following the financial crisis due in part to questionable lending practices.

## Multifamily Securitization Program Data
**Data Source: ** [Freddie Mac Investor Access](https://msia.ficonsulting.com/)

The complete data set as of 2/22/2017 is in: 'data/custom_rpt_all_properties_20170222.csv'

This data is for all loans that Freddie Mac has securitized. The process of securitization for multifamily loans involves lumping together multiple mortgages across multiple properties into a pool of mortgages. This pool is then divided up into individual securities that are sold to investors.

Freddie Mac issues securities in 12 deal types:

| **Deal Type**        | **Descriptor** | **Description**                       | **# of Deals** | **Total UPB**    |
| -------------------- |:--------------:| ------------------------------------- |:--------------:| ----------------:|   
| 10 Year              | K-000          | fixed loans with mostly 10 year terms | 55             | $75,679,377,782  |
| 7 Year               | K-700          | fixed loans with 7 year terms         | 23             | $29,259,691,719  |
| 5 Year               | K-500          | fixed loans with 5 year terms         | 4              | $4,106,237,917   |
| Single Sponsor       | K-ABC          | single sponsor, sometimes single asset| 14             | $14,792,788,254  |
| No-Subordination     | K-P00          | portfolio loans, no subordinate piece | 3              | $2,622,223,362   |
| Floating Rate        | K-F00          | floating rates of various terms       | 21             | $28,429,449,460  |
| Seniors Housing      | K-S00          | senior multifamily mortgages          | 5              | $3,832,996,970   |
| Seasoned             | K-X00          | seasoned loans                        | 2              | $1,107,896,608   |
| Supplemental         | K-J00          | supplemental loans                    | 7              | $1,593,302,089   |
| >10 Year             | K-1500         | fixed loans, greater than 10 year term| 2              | $1,194,204,846   |
| Small Balance        | SB-00          | small balance, also known as FRESB    | 21             | $4,563,707,762   |
| Workforce            | K-W00          | workforce housing loans 55            | 1              | $676,185,705     |
|                      |                |                                       |      Total UPB | $167,858,062,473 |
1 As of September 30, 2016                                                  
2 Excludes Q-Deals

**Pros / Cons:**

Pros - this dataset covers many

For the initial analysis, only the 10 Year K-000 series deals will be analyzed. These deals span the longest period of longevity, which is from 2009-2017.

| **Column Name**      | **Type** | **Non-null**     | **Unique** |
| -------------------- |:------------:|:------------:|:-----------:|
|  #_properties | object | 7855 | 8  |
|  2nd_preceding_fydscr_(ncf) | object | 7855 | 1356  |
|  actual_balance | object | 7855 | 6650  |
|  balance_range | object | 7855 | 13  |
|  balloon | object | 7855 | 2  |
|  city | object | 7855 | 1771  |
|  current%_of_deal | float64 | 7855 | 147  |
|  current_bal_rank | int64 | 7855 | 153  |
|  current_endingsched_balance | object | 7855 | 6650  |
|  currentloan_balance | object | 7855 | 6650  |
|  cut-off_dateloan_balance | object | 7855 | 6007  |
|  date_added_to_servicer_watchlist_ | object | 7855 | 94  |
|  deal | object | 7855 | 113  |
|  distributiondate | object | 7855 | 1  |
|  dscr_(ncf) | float64 | 7855 | 1765  |
|  dscr_range | object | 7855 | 13  |
|  dlq._status | object | 7855 | 7  |
|  dlq_status_text | object | 7855 | 7  |
|  fka_status_of_loan | object | 7855 | 7  |
|  group_id | object | 7855 | 10  |
|  int._only | object | 7855 | 3  |
|  loan_amortization_type | object | 7855 | 9  |
|  loan_id | int64 | 7855 | 7780  |
|  ltv_range | object | 7855 | 8  |
|  master_servicer | object | 7855 | 6  |
|  maturity_date | object | 7855 | 228  |
|  most_recentdscr_(ncf) | object | 7855 | 1682  |
|  most_recent_financial_'as_of'_end_date_ | object | 7855 | 22  |
|  most_recent_financial_'as_of'_start_date_ | object | 7855 | 24  |
|  most_recentfinancial_indicator | object | 7855 | 7  |
|  most_recentmaster_servicerreturn_date_ | object | 7855 | 8  |
|  most_recent_ncf | object | 7855 | 5605  |
|  most_recent_noi | object | 7855 | 5597  |
|  most_recentphys_occup | object | 7855 | 80  |
|  most_recentspecial_servicertransfer_date_ | object | 7855 | 9  |
|  most_recent_value | object | 7855 | 2324  |
|  mps_deal_alias | object | 7855 | 113  |
|  msa | object | 7855 | 405  |
|  no_time_dlq12mth | int64 | 7855 | 5  |
|  no_time_dlqlife | int64 | 7855 | 11  |
|  note_rate | object | 7855 | 582  |
|  note_rate_range | object | 7855 | 13  |
|  occupancy | float64 | 7855 | 9  |
|  occupancy_date | object | 7855 | 503  |
|  occupancy_range | object | 7855 | 7  |
|  occupancy_source | object | 7855 | 4  |
|  operatingtrust_advisor | object | 11 | 2  |
|  orig_amortization | object | 7855 | 31  |
|  orig_term | object | 7855 | 78  |
|  original_note_amount | object | 7855 | 4241  |
|  original_occupancy_date | object | 7855 | 1290  |
|  original_occupancy__rate | float64 | 7855 | 788  |
|  paid_thru_date | object | 7855 | 11  |
|  payment_freq | object | 7855 | 2  |
|  preceding_fiscalyear_financial_'as_of'_date_ | object | 7855 | 27  |
|  preceding_fy_ncf | object | 7855 | 6205  |
|  preceding_fiscalyear_noi | object | 7855 | 6201  |
|  preceding_fy_dscr_(ncf) | object | 7855 | 1643  |
|  property_name | object | 7855 | 7296  |
|  property_subtype | object | 7855 | 43  |
|  prospectus_id | int64 | 7855 | 155  |
|  remaining_term | object | 7855 | 218  |
|  seasoning | object | 7855 | 105  |
|  seasoning_range | object | 7855 | 12  |
|  second_preceding_fiscal_year_financial_'as_of'_date_ | object | 7855 | 23  |
|  second_preceding_fy_ncf_(dscr) | object | 7855 | 1357  |
|  second_preceding_fy_ncf | object | 7855 | 4070  |
|  second_precedingfiscal_year_noi | object | 7855 | 4065  |
|  servicer_watchlistcode | object | 7855 | 38  |
|  special_servicer | object | 7855 | 17  |
|  state | object | 7855 | 50  |
|  total_reserve_bal | object | 7855 | 4499  |
|  uw_dscr_(ncf)amortizing | float64 | 7855 | 686  |
|  uw_ltv | object | 7855 | 569  |
|  year_built | object | 7855 | 215  |
|  zip_code | object | 7855 | 3599  |
|  property_address | object | 7855 | 7582  |
|  property_city | object | 7855 | 1861  |
|  revenue_at_contribution | object | 7855 | 3926  |
|  operating_expenses_at_contribution | object | 7855 | 4573  |
|  noi_at_contribution | object | 7855 | 6250  |
|  dscr_(noi)_at_contribution | object | 7855 | 1047  |
|  ncf_at_contribution | object | 7855 | 5993  |
|  dscr_(ncf)_at_contribution | object | 7855 | 496  |
|  second_preceding_fy_revenue | object | 7855 | 4067  |
|  second_preceding_fy_operating_exp | object | 7855 | 4067  |
|  second_preceding_fy_debt_serv_amt | object | 7855 | 4061  |
|  preceding_fy_revenue | object | 7855 | 6196  |
|  preceding_fy_operating_exp | object | 7855 | 6210  |
|  preceding_fy_debt_svc_amount | object | 7855 | 6202  |
|  most_recent_revenue | object | 7855 | 5605  |
|  most_recent_operating_expenses | object | 7855 | 5606  |
|  most_recent_debt_service_amount | object | 7855 | 5604  |
