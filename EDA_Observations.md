# Observations of Data During EDA

## Multifamily Loan Performance Dataset (MFLP) - mlpd_datamart_1q16.txt
**Data Source: ** [Investor Tools - Loan Performance Database](http://www.freddiemac.com/multifamily/investors/reporting.html)

The Database provides historical information on a subset of the Freddie Mac Multifamily whole loan portfolio since 1994.  It includes information on original loan terms; identifiers for prepaid loans, defaulted loans and delinquencies; property information; and dates of real estate owned (REO) sales.

There are 338,445 rows of data covering 11,570 unique loans. Multiple loans may be related to the same property.

In [21]: df['lnno'].nunique()  
Out[21]: 11570

**Column Data Types:**

|  Column Name | Type | Non-null | Unique | Example  | **Convert** |
|  :--- | :---: | :---: | :---: | :---  | :-----------:|
|  lnno | int64 | 338445 | 11570 | 99940  | leave       |
|  quarter | object | 338445 | 89 | y07q3  | datetime    |
|  mrtg_status | int64 | 338445 | 5 | 100  | int64 -> get_dummies |
|  amt_upb_endg | float64 | 328999 | 263138 | 5659645.77  |  leave      |
|  liq_dte | object | 9160 | 2332 | nan  | datetime    |
|  liq_upb_amt | float64 | 9164 | 8586 | nan  | leave       |
|  dt_sold | object | 73 | 60 | nan  | datetime    |
|  cd_fxfltr | object | 130063 | 3 | FXDFLT  | str -> get_dummies |
|  cnt_amtn_per | float64 | 338428 | 148 | 360.0  | leave       |
|  cnt_blln_term | float64 | 338428 | 220 | 87.0  | leave       |
|  cnt_io_per | float64 | 338419 | 105 | 0.0  | leave       |
|  dt_io_end | object | 98108 | 225 | nan  | datetime    |
|  cnt_mrtg_term | int64 | 338445 | 243 | 87  | leave       |
|  cnt_rsdntl_unit | float64 | 338411 | 768 | 336.0  | leave       |
|  cnt_yld_maint | int64 | 338445 | 178 | 75  | leave       |
|  code_int | object | 338445 | 2 | FIX  | str -> get_dummies |
|  rate_dcr | float64 | 338290 | 2015 | 1.311  | leave       |
|  rate_int | float64 | 338416 | 7296 | 0.0594  | leave       |  
|  rate_ltv | float64 | 338392 | 6905 | 0.79122  | leave       |
|  amt_upb_pch | float64 | 338445 | 4659 | 5669285.36  | leave       |
|  dt_fund | object | 338445 | 3589 | 25JUL2007  | datetime    |
|  dt_mty | object | 338445 | 484 | 01SEP2014  | datetime    |
|  code_st | object | 338445 | 50 | LA  | leave       |
|  geographical_region | object | 338048 | 549 | New Orleans, LA MSA  | leave       |
|  id_link_grp | float64 | 66308 | 1118 | 75103.0  | leave       |
|  code_sr | object | 19684 | 4 | nan  | str -> get_dummies |
|  reo_operating_expinc | float64 | 74 | 70 | nan  | leave       |
|  prefcl_fcl_expinc | float64 | 74 | 73 | nan  | leave       |
|  selling_expinc | float64 | 74 | 69 | nan  | leave       |
|  sales_price | float64 | 73 | 65 | nan  | leave       |

**'id_link_grp'** links together loans that are related to the same property

**Pros / Cons**
Pros - This dataset is well compiled and would require little data cleaning and adjustments before running through models.

Cons - This is a subset of the overall Freddie Mac multifamily loans issued and may not be representative of the typical loan funded by Freddie Mac and sold to investors. These are only the loans of which Freddie Mac has retained ownership. There could be a variety of reasons that they would have retained ownership of these loans. These loans could be for types of properties that are not easily bundled with other properties for securitization. This could apply to senior housing properties with some level of care (i.e. assisted living, nursing homes, memory care, etc.). This could also apply to affordable housing projects with complex capital structures and mechanisms that would potentially deter the secondary capital markets from investing in their securities. A few examples of some characteristics that could deter capital markets would be rent restrictions, phasing off tax abatements, land leases, or rent subsidies from entities with low credit ratings, such as, a bankrupt county or city. I would Freddie Mac not cherry pick these loans to inflate their performance data. The federal government did take control of both Freddie Mac and Fannie Mae following the financial crisis due in part to questionable lending practices.

## Multifamily Securitization Program Data (MSPD)
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

The data available for the securitized loans is as follows:

|  Column Name | Type | Non-null | Unique | Example  | Convert | Drop |
|  :--- | :---: | :---: | :---: | :---  | :---: | :---: |
|  #_properties | object | 7855 | 8 | 1  | int64 | Y |
|  2nd_preceding_fydscr_(ncf) | object | 7855 | 1356 | -  | float64 | Y |
|  actual_balance | object | 7855 | 6650 | $878,000,000  | float64 | N |
|  balance_range | object | 7855 | 13 | 20.00+  | str -> get_dummies | N |
|  balloon | object | 7855 | 2 | Y  | str -> get_dummies | Y |
|  city | object | 7855 | 1771 | Los Angeles  | str | Y |
|  current%_of_deal | float64 | 7855 | 147 | 100.0  | leave | Y |
|  current_bal_rank | int64 | 7855 | 153 | 1  | leave | Y |
|  current_endingsched_balance | object | 7855 | 6650 | $878,000,000  | float64 | Y |
|  currentloan_balance | object | 7855 | 6650 | $878,000,000  | float64 | Y |
|  cut-off_dateloan_balance | object | 7855 | 6007 | $878,000,000  | float64 | Y |
|  date_added_to_servicer_watchlist_ | object | 7855 | 94 | -  | datetime | N |
|  deal | object | 7855 | 113 | FREMF 2015-KPLB  | str -> get_dummies | N |
|  distributiondate | object | 7855 | 1 | 01/25/2017  | datetime | Y |
|  dscr_(ncf) | float64 | 7855 | 1765 | 2.74  | leave | N |
|  dscr_range | object | 7855 | 13 | 2.00 and up  | str -> get_dummies | N |
|  dlq._status | object | 7855 | 7 | Current  | str -> get_dummies | Y |
|  dlq_status_text | object | 7855 | 7 | Current  | str -> get_dummies | N |
|  fka_status_of_loan | object | 7855 | 7 | 0  | str -> get_dummies | N |
|  group_id | object | 7855 | 10 | -  | str -> get_dummies | N |
|  int._only | object | 7855 | 3 | Y  | str -> get_dummies | Y |
|  loan_amortization_type | object | 7855 | 9 | Interest Only  | str -> get_dummies | N |
|  loan_id | int64 | 7855 | 7780 | 708125468  | leave | N |
|  ltv_range | object | 7855 | 8 | 50.0% - 55.0%  | str -> get_dummies | N |
|  master_servicer | object | 7855 | 6 | Freddie Mac  | str -> get_dummies | N |
|  maturity_date | object | 7855 | 228 | 05/01/2025  | datetime | Y |
|  most_recentdscr_(ncf) | object | 7855 | 1682 | 2.740  | float64 | N |
|  most_recent_financial_'as_of'_end_date_ | object | 7855 | 22 | 09/30/2016  | datetime | Y |
|  most_recent_financial_'as_of'_start_date_ | object | 7855 | 24 | 01/01/2016  |  datetime | Y |
|  most_recentfinancial_indicator | object | 7855 | 7 | Current  | str -> get_dummies | N |
|  most_recentmaster_servicerreturn_date_ | object | 7855 | 8 | -  | datetime | N |
|  most_recent_ncf | object | 7855 | 5605 | $61,193,871  | float64 | N |
|  most_recent_noi | object | 7855 | 5597 | $61,830,621  | float64 | N |
|  most_recentphys_occup | object | 7855 | 80 | 97.0  | float64 | N |
|  most_recentspecial_servicertransfer_date_ | object | 7855 | 9 | -  | datetime | N |
|  most_recent_value | object | 7855 | 2324 | $1,675,000,000  | float64 | N |
|  mps_deal_alias | object | 7855 | 113 | 2015-KPLB  | str | N |
|  msa | object | 7855 | 405 | Los Angeles-Long Beach-Santa Ana CA  | str -> get_dummies | N |
|  no_time_dlq12mth | int64 | 7855 | 5 | 0  | str -> get_dummies | N |
|  no_time_dlqlife | int64 | 7855 | 11 | 0  | str -> get_dummies | N |
|  note_rate | object | 7855 | 582 | 3.3300  | float64 | N |
|  note_rate_range | object | 7855 | 13 | 3.001% - 3.500%  | str -> get_dummies | N |
|  occupancy | float64 | 7855 | 9 | 1.0  | leave | N |
|  occupancy_date | object | 7855 | 503 | 09/30/2016  | datetime | Y |
|  occupancy_range | object | 7855 | 7 | > 95.0%  | str -> get_dummies | N |
|  occupancy_source | object | 7855 | 4 | MOST RECENT  | str | Y |
|  operatingtrust_advisor | object | 11 | 2 | Freddie Mac  | str -> get_dummies | Y |
|  orig_amortization | object | 7855 | 31 | 0  | int64 | N |
|  orig_term | object | 7855 | 78 | 120  | int64 | N |
|  original_note_amount | object | 7855 | 4241 | $878,000,000  | float64 | N |
|  original_occupancy_date | object | 7855 | 1290 | 03/17/2015  | datetime | Y |
|  original_occupancy__rate | float64 | 7855 | 788 | 97.7  | float64 | N |
|  paid_thru_date | object | 7855 | 11 | 01/01/2017  | datetime | Y |
|  payment_freq | object | 7855 | 2 | Monthly  | str -> drop | Y |
|  preceding_fiscalyear_financial_'as_of'_date_ | object | 7855 | 27 | 12/31/2015  | datetime | Y |
|  preceding_fy_ncf | object | 7855 | 6205 | $76,957,036  | float64 | Y |
|  preceding_fiscalyear_noi | object | 7855 | 6201 | $77,806,036  | float64 | Y |
|  preceding_fy_dscr_(ncf) | object | 7855 | 1643 | 2.600  | float64 | Y |
|  property_name | object | 7855 | 7296 | Park La Brea Apartments  | str -> drop | Y |
|  property_subtype | object | 7855 | 43 | Garden and High Rise  | str -> clean -> get_dummies | N |
|  prospectus_id | int64 | 7855 | 155 | 1  | str -> drop | Y |
|  remaining_term | object | 7855 | 218 | 100  | int64 | Y |
|  seasoning | object | 7855 | 105 | 20  | int64 | Y |
|  seasoning_range | object | 7855 | 12 | < 14  | str -> fix/drop | Y |
|  second_preceding_fiscal_year_financial_'as_of'_date_ | object | 7855 | 23 | -  | datetime | Y |
|  second_preceding_fy_ncf_(dscr) | object | 7855 | 1357 | -  | float64 | Y |
|  second_preceding_fy_ncf | object | 7855 | 4070 | $0  | float64 | Y |
|  second_precedingfiscal_year_noi | object | 7855 | 4065 | $0  | float64 | Y |
|  servicer_watchlistcode | object | 7855 | 38 | -  | str -> investigate | Y |
|  special_servicer | object | 7855 | 17 | Berkeley Point Capital, LLC  | str -> get_dummies | N |
|  state | object | 7855 | 50 | CA  | str -> get_dummies | N |
|  total_reserve_bal | object | 7855 | 4499 | 0  | float64 | Y |
|  uw_dscr_(ncf)amortizing | float64 | 7855 | 686 | 2.27  | float64 | N |
|  uw_ltv | object | 7855 | 569 |  52.4%  | float64 | N |
|  year_built | object | 7855 | 215 | 1944 - 1951  | str -> clean -> get_dummies | N |
|  zip_code | object | 7855 | 3599 | 90036  | str -> clean | Y |
|  property_address | object | 7855 | 7582 | 6200 West Third Street  | str -> drop | Y |
|  property_city | object | 7855 | 1861 | Los Angeles  | str -> drop | Y |
|  revenue_at_contribution | object | 7855 | 3926 | 105197374  | float64 | Y |
|  operating_expenses_at_contribution | object | 7855 | 4573 | 37126374  | float64 | Y |
|  noi_at_contribution | object | 7855 | 6250 | 68071000  | float64 | Y |
|  dscr_(noi)_at_contribution | object | 7855 | 1047 | -  | float64 | Y |
|  ncf_at_contribution | object | 7855 | 5993 | 67222000  | float64 | Y |
|  dscr_(ncf)_at_contribution | object | 7855 | 496 | 2.27  | float64 | Y |
|  second_preceding_fy_revenue | object | 7855 | 4067 | 0  | float64 | Y |
|  second_preceding_fy_operating_exp | object | 7855 | 4067 | 0  | float64 | Y |
|  second_preceding_fy_debt_serv_amt | object | 7855 | 4061 | -  | float64 | Y |
|  preceding_fy_revenue | object | 7855 | 6196 | 108541632.46  | float64 | Y |
|  preceding_fy_operating_exp | object | 7855 | 6210 | 30735596.87  | float64 | Y |
|  preceding_fy_debt_svc_amount | object | 7855 | 6202 | 29643475  | float64 | Y |
|  most_recent_revenue | object | 7855 | 5605 | 85254239.66  | float64 | Y |
|  most_recent_operating_expenses | object | 7855 | 5606 | 23423618.25  | float64 | Y |
|  most_recent_debt_service_amount | object | 7855 | 5604 | 22334125  | float | Y |

**Pros / Cons:**

Pros - This dataset covers all securitized loans. It appears to include not just active, but previously foreclosed loans as well.

Cons - This data does not contain quarterly time-series data for each property, but does contain the property financial performance data at origination (loan creation), most recent of quarter, most recent end-of-year, and previous year end-of year data. This covers about three years of a loan's term. The longest allowable loan term is 10 years and most get refinanced prior to the maturity date or have existence for significantly less that 10 years, so covering 3 years of the term should give an indication of changes in performance.

## Dataset for Analysis

Data will be extracted from both multifamily loan datasets (MFLP and MSPD). Common columns will be combined into one larger dataset. One main benefit to this is increasing the data set to the combined set of 19,425 mortgages (7,855 securitized and 11,570 Freddie held mortgages).

The features that will be used are:

| Column Name     | MFLP Column Name | MSPD Column Name |
| -----------     | ---------------- | ---------------- |
| loan_id         | lnno             | loan_id          |
| loan_status*    | mrtg_status      | dlq_status_text  |
| current_balance | amt_upb_endg     | actual_balance   |
| property_state  | code_st          | state            |
| int_rate        | rate_int         | note_rate        |
| dsc_ratio       | rate_dcr         | dscr_(ncf)       |

** * label derived from
