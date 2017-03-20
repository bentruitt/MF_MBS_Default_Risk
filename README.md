# Analysis of Multifamily Mortgage Default Probabilities

## Project Introduction
The focus of this analysis is to approximate the probability of default for multifamily mortgages. I have worked in the area of multifamily lending for about 10 years. I have done everything from underwrite to originate to the sale of securities to large investment banks that would re-securitize the loans for consumers such as large institutional investors and hedge funds.

<img align="right" src="/img/thumbs_1600-Glenarm-002.jpg" alt="1600 Glenarm Place" width=40%>

The loan data that I will be using for this analysis represents the performance of Freddie Mac multifamily loans. Freddie Mac is the nickname used for the Federal Home Loan Mortgage Corporation (FHLMC). Freddie Mac works in partnership with a network of nationwide lenders to provide loans for both single-family and multifamily mortgages. A single-family property is defined as 1-4 units. A single-family mortgage is what you think of if you were to get a mortgage on your home. You may already have a Freddie Mac mortgage and not know it. A multifamily mortgage is what is provided for an apartment building or complex. A multifamily property is 5 units or more. An example of a multifamily property is 1600 Glenarm Place in Denver, CO. The focus of this analysis will be only multifamily mortgages.

## Freddie Mac

### U.S. Government Implicit Guarantee

<img align="right" src="/img/investor-presentation.pdf.jpg" alt="Freddie Fed Bailout" width=60%>

During the financial crisis in 2008, Freddie Mac received a bailout from the U.S. Government due to defaulting mortgages in their portfolio. Freddie Mac was given a loan for $71.3 Billion. Since this time, Freddie Mac has repaid $105.9 Billion in dividends. When the U.S. Government bailed-out Freddie Mac, the statement was made that Freddie Mac was "too big to fail" and had the implicit guarantee of the U.S. Government. Investors now began buying Freddie Mac mortgage securities as if they had no default risk. This reduced the interest rate required by investors to less than what would be required if the loan were made by a normal bank.

As of March 7th, 2017; if you look at the current interest rate required for a loan by Freddie Mac in comparison to a conventional / bank loan for a mortgage with a payment based on 30 years of amortization, term of 10 years, loan-to-value ratio of 80%, and a mortgage amount of $10,000,000, the rates would be:

* Freddie Mac Loan: 4.450%
* Conventional Loan: 4.711%

This does not look like a significant difference; however, let's assume you were the one investing the $10,000,000 to provide this loan and you did it based on Freddie Mac's default free status. Now the federal government announces that they will no longer provide their implicit guarantee. Your $10,000,000 investment now drops to a value of $9,809,682 a difference of $190,318 (or 1.90%). This is approximately the value associated with the default risk priced into the conventional loan. This is the value associated with the U.S. Government's implicit guarantee.

### Why Freddie Mac

<img align="right" src="/img/freddie-mac-to-pay-treasury-4-5-billion-after-reporting-profit.jpg" alt="Freddie Fed Big Payment" width=50%>

The political climate and public opinion has shifted. Freddie Mac appears to be standing on their own two feet and the implicit guarantee does not seem as necessary, in particular, now that Freddie Mac has paid back their loan. It is quite possible that the U.S. Government's implicit guarantee could be removed.

If the U.S. Government's implicit guarantee were to be removed it could significantly effect the value of Freddie Mac issued securities. This would be good for some and bad for others, but getting ahead of this by having a good understanding of the current default probabilities within the Freddie Mac loan portfolio and how to forecast the values that the securities would likely correct to is key in creating and profiting from the opportunities that will surely arise.

## Consumers of This Model

### Freddie Mac Security Investors
Two of the driving factors for investors to invest in Freddie Mac (and other GSE) securities are their ability to be leveraged, such as by hedge funds, and their ability to meet the Dodd-Frank Act requirement that banks and insurance companies hold higher percentages of risk free assets on their balance sheets.

The Dodd-Frank Act requirement is by far the largest domestic driver for the purchase of Freddie Mac securities. If a bank wants to increase their lending, they need to increase the amount of risk free assets on their balance sheets. If an insurance company wants to increase their liabilities, by issuing new policies, they need to hold more risk free assets. This aspect of the Dodd-Frank act is hendering the profitability of banks and insurance companies and they want it gone. They have been steadily lobbying for a change to this portion of the Dodd-Frank act and it may just be around the corner.

Hedge funds create leverage in their portfolios by leveraging their purchases. Leveraging is just another term for making purchases using a loan from somebody else. The purchases of risk free assets can be leveraged to a higher ratio of around 1-to-4 than purchases of assets that have a higher risk of default, such as, Commercial Mortgage Backed Security (CMBS) assets which may be leverage to around 1-to-3. If you look back at our Freddie Mac interest rate of 4.450%. A hedge fund may pay around 2.5% to borrow money, so after leverage they are making a return of just over 10% for the Freddie Mac loan investment and just under 8% for the conventional loan investment.

There are many likely to be affected by a removal of the U.S. Government's implicit guarantee, but a few that should be sure to pay particular attention to my results are insurance companies, banks, and hedge funds.

**Link to Presentation Slides:** [Default Risk In Multifamily Mortgages](https://docs.google.com/presentation/d/1AvFxeSGNUIpF76LP149ydiUKDZK0Xt6j5LOWwd6UNew/pub?start=true&loop=true&delayms=3000)

# Teaser
<img align="center" src="/plots/web/GradientBoostingClassifier_default_prob_hist_f7_2017-03-14_13:11.png" alt="Default Probabilities" width=75%>

<img align="center" src="/plots/web/GradientBoostingClassifier_ROC_plot_f7_2017-03-14_13:11.png" alt="ROC Plot" width=75%>

<img align="center" src="/plots/web/GradientBoostingClassifier_Conf_Matrix_f7_2017-03-14_13:11.png" alt="Confusion Matrix" width=75%>

# Observations of Data During EDA

## Multifamily Loan Performance Dataset (MFLP) - mlpd_datamart_1q16.txt
**Data Source: **  
[Investor Tools - Loan Performance Database](http://www.freddiemac.com/multifamily/investors/reporting.html)

The Database provides historical information on a subset of the Freddie Mac Multifamily whole loan portfolio since 1994.  It includes information on original loan terms; identifiers for prepaid loans, defaulted loans and delinquencies; property information; and dates of real estate owned (REO) sales.

There are 338,445 rows of data covering 11,570 unique loans. Multiple loans may be related to the same property.

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
|  id_link_grp* | float64 | 66308 | 1118 | 75103.0  | leave       |
|  code_sr | object | 19684 | 4 | nan  | str -> get_dummies |
|  reo_operating_expinc | float64 | 74 | 70 | nan  | leave       |
|  prefcl_fcl_expinc | float64 | 74 | 73 | nan  | leave       |
|  selling_expinc | float64 | 74 | 69 | nan  | leave       |
|  sales_price | float64 | 73 | 65 | nan  | leave       |

** * 'id_link_grp' links together loans that are related to the same property

**Pros / Cons**  
Pros - This dataset is well compiled and would require little data cleaning and adjustments before running through models.

Cons - This is a subset of the overall Freddie Mac multifamily loans issued and may not be representative of the typical loan funded by Freddie Mac and sold to investors. These are only the loans of which Freddie Mac has retained ownership. There could be a variety of reasons that they would have retained ownership of these loans. These loans could be for types of properties that are not easily bundled with other properties for securitization. This could apply to senior housing properties with some level of care (i.e. assisted living, nursing homes, memory care, etc.). This could also apply to affordable housing projects with complex capital structures and mechanisms that would potentially deter the secondary capital markets from investing in their securities. A few examples of some characteristics that could deter capital markets would be rent restrictions, phasing off tax abatements, land leases, or rent subsidies from entities with low credit ratings, such as, a bankrupt county or city. I would Freddie Mac not cherry pick these loans to inflate their performance data. The federal government did take control of both Freddie Mac and Fannie Mae following the financial crisis due in part to questionable lending practices.

## Multifamily Securitization Program Data (MSPD)
**Data Source: **  
[Freddie Mac Investor Access](https://msia.ficonsulting.com/)

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

|  Column Name | Type | Non-null | Unique | Example  | Convert |
|  :--- | :---: | :---: | :---: | :---  | :---: |
|  #_properties | object | 7855 | 8 | 1  | int64 |
|  2nd_preceding_fydscr_(ncf) | object | 7855 | 1356 | -  | float64 |
|  actual_balance | object | 7855 | 6650 | $878,000,000  | float64 |
|  balance_range | object | 7855 | 13 | 20.00+  | str -> get_dummies |
|  balloon | object | 7855 | 2 | Y  | str -> get_dummies |
|  city | object | 7855 | 1771 | Los Angeles  | str |
|  current%_of_deal | float64 | 7855 | 147 | 100.0  | leave |
|  current_bal_rank | int64 | 7855 | 153 | 1  | leave |
|  current_endingsched_balance | object | 7855 | 6650 | $878,000,000  | float64 |
|  currentloan_balance | object | 7855 | 6650 | $878,000,000  | float64 |
|  cut-off_dateloan_balance | object | 7855 | 6007 | $878,000,000  | float64 |
|  date_added_to_servicer_watchlist_ | object | 7855 | 94 | -  | datetime |
|  deal | object | 7855 | 113 | FREMF 2015-KPLB  | str -> get_dummies |
|  distributiondate | object | 7855 | 1 | 01/25/2017  | datetime |
|  dscr_(ncf) | float64 | 7855 | 1765 | 2.74  | leave |
|  dscr_range | object | 7855 | 13 | 2.00 and up  | str -> get_dummies |
|  dlq._status | object | 7855 | 7 | Current  | str -> get_dummies |
|  dlq_status_text | object | 7855 | 7 | Current  | str -> get_dummies |
|  fka_status_of_loan | object | 7855 | 7 | 0  | str -> get_dummies |
|  group_id | object | 7855 | 10 | -  | str -> get_dummies |
|  int._only | object | 7855 | 3 | Y  | str -> get_dummies |
|  loan_amortization_type | object | 7855 | 9 | Interest Only  | str -> get_dummies |
|  loan_id | int64 | 7855 | 7780 | 708125468  | leave |
|  ltv_range | object | 7855 | 8 | 50.0% - 55.0%  | str -> get_dummies |
|  master_servicer | object | 7855 | 6 | Freddie Mac  | str -> get_dummies |
|  maturity_date | object | 7855 | 228 | 05/01/2025  | datetime |
|  most_recentdscr_(ncf) | object | 7855 | 1682 | 2.740  | float64 |
|  most_recent_financial_'as_of'_end_date_ | object | 7855 | 22 | 09/30/2016  | datetime |
|  most_recent_financial_'as_of'_start_date_ | object | 7855 | 24 | 01/01/2016  |  datetime |
|  most_recentfinancial_indicator | object | 7855 | 7 | Current  | str -> get_dummies |
|  most_recentmaster_servicerreturn_date_ | object | 7855 | 8 | -  | datetime |
|  most_recent_ncf | object | 7855 | 5605 | $61,193,871  | float64 |
|  most_recent_noi | object | 7855 | 5597 | $61,830,621  | float64 |
|  most_recentphys_occup | object | 7855 | 80 | 97.0  | float64 |
|  most_recentspecial_servicertransfer_date_ | object | 7855 | 9 | -  | datetime |
|  most_recent_value | object | 7855 | 2324 | $1,675,000,000  | float64 |
|  mps_deal_alias | object | 7855 | 113 | 2015-KPLB  | str |
|  msa | object | 7855 | 405 | Los Angeles-Long Beach-Santa Ana CA  | str -> get_dummies |
|  no_time_dlq12mth | int64 | 7855 | 5 | 0  | str -> get_dummies |
|  no_time_dlqlife | int64 | 7855 | 11 | 0  | str -> get_dummies |
|  note_rate | object | 7855 | 582 | 3.3300  | float64 |
|  note_rate_range | object | 7855 | 13 | 3.001% - 3.500%  | str -> get_dummies |
|  occupancy | float64 | 7855 | 9 | 1.0  | leave |
|  occupancy_date | object | 7855 | 503 | 09/30/2016  | datetime |
|  occupancy_range | object | 7855 | 7 | > 95.0%  | str -> get_dummies |
|  occupancy_source | object | 7855 | 4 | MOST RECENT  | str |
|  operatingtrust_advisor | object | 11 | 2 | Freddie Mac  | str -> get_dummies |
|  orig_amortization | object | 7855 | 31 | 0  | int64 |
|  orig_term | object | 7855 | 78 | 120  | int64 |
|  original_note_amount | object | 7855 | 4241 | $878,000,000  | float64 |
|  original_occupancy_date | object | 7855 | 1290 | 03/17/2015  | datetime |
|  original_occupancy__rate | float64 | 7855 | 788 | 97.7  | float64 |
|  paid_thru_date | object | 7855 | 11 | 01/01/2017  | datetime |
|  payment_freq | object | 7855 | 2 | Monthly  | str -> drop |
|  preceding_fiscalyear_financial_'as_of'_date_ | object | 7855 | 27 | 12/31/2015  | datetime |
|  preceding_fy_ncf | object | 7855 | 6205 | $76,957,036  | float64 |
|  preceding_fiscalyear_noi | object | 7855 | 6201 | $77,806,036  | float64 |
|  preceding_fy_dscr_(ncf) | object | 7855 | 1643 | 2.600  | float64 |
|  property_name | object | 7855 | 7296 | Park La Brea Apartments  | str -> drop |
|  property_subtype | object | 7855 | 43 | Garden and High Rise  | str -> clean -> get_dummies |
|  prospectus_id | int64 | 7855 | 155 | 1  | str -> drop |
|  remaining_term | object | 7855 | 218 | 100  | int64 |
|  seasoning | object | 7855 | 105 | 20  | int64 |
|  seasoning_range | object | 7855 | 12 | < 14  | str -> fix/drop |
|  second_preceding_fiscal_year_financial_'as_of'_date_ | object | 7855 | 23 | -  | datetime |
|  second_preceding_fy_ncf_(dscr) | object | 7855 | 1357 | -  | float64 |
|  second_preceding_fy_ncf | object | 7855 | 4070 | $0  | float64 |
|  second_precedingfiscal_year_noi | object | 7855 | 4065 | $0  | float64 |
|  servicer_watchlistcode | object | 7855 | 38 | -  | str -> investigate |
|  special_servicer | object | 7855 | 17 | Berkeley Point Capital, LLC  | str -> get_dummies |
|  state | object | 7855 | 50 | CA  | str -> get_dummies |
|  total_reserve_bal | object | 7855 | 4499 | 0  | float64 |
|  uw_dscr_(ncf)amortizing | float64 | 7855 | 686 | 2.27  | float64 |
|  uw_ltv | object | 7855 | 569 |  52.4%  | float64 |
|  year_built | object | 7855 | 215 | 1944 - 1951  | str -> clean -> get_dummies |
|  zip_code | object | 7855 | 3599 | 90036  | str -> clean |
|  property_address | object | 7855 | 7582 | 6200 West Third Street  | str -> drop |
|  property_city | object | 7855 | 1861 | Los Angeles  | str -> drop |
|  revenue_at_contribution | object | 7855 | 3926 | 105197374  | float64 |
|  operating_expenses_at_contribution | object | 7855 | 4573 | 37126374  | float64 |
|  noi_at_contribution | object | 7855 | 6250 | 68071000  | float64 |
|  dscr_(noi)_at_contribution | object | 7855 | 1047 | -  | float64 |
|  ncf_at_contribution | object | 7855 | 5993 | 67222000  | float64 |
|  dscr_(ncf)_at_contribution | object | 7855 | 496 | 2.27  | float64 |
|  second_preceding_fy_revenue | object | 7855 | 4067 | 0  | float64 |
|  second_preceding_fy_operating_exp | object | 7855 | 4067 | 0  | float64 |
|  second_preceding_fy_debt_serv_amt | object | 7855 | 4061 | -  | float64 |
|  preceding_fy_revenue | object | 7855 | 6196 | 108541632.46  | float64 |
|  preceding_fy_operating_exp | object | 7855 | 6210 | 30735596.87  | float64 |
|  preceding_fy_debt_svc_amount | object | 7855 | 6202 | 29643475  | float64 |
|  most_recent_revenue | object | 7855 | 5605 | 85254239.66  | float64 |
|  most_recent_operating_expenses | object | 7855 | 5606 | 23423618.25  | float64 |
|  most_recent_debt_service_amount | object | 7855 | 5604 | 22334125  | float |

**Pros / Cons:**

Pros - This dataset covers all securitized loans. It includes not just active, but previously foreclosed loans as well.

Cons - This data does not contain quarterly time-series data for each property, but does contain the property financial performance data at origination (loan creation), most recent of quarter, most recent end-of-year, and previous year end-of year data. This covers about three years of a loan's term. The longest allowable loan term is 10 years and most get refinanced prior to the maturity date or have existence for significantly less that 10 years, so covering 3 years of the term should give an indication of changes in performance.

## Dataset for Analysis

After exploratory data analysis it was concluded that the MSPD dataset contained more features of potential value and also allowed for more insightful engineered features to be constructed. The following features were extracted or engineered from the MSPD dataset:

|  Column Name | Type | Non-null | Unique | Example  | Engineering |
|  :--- | :---: | :---: | :---: | :---  |:---:|
|  label | float64 | 7855 | 2 | 0.0  | YES* |
|  loan_id | int64 | 7855 | 7780 | 708125468  | NO |
|  current_balance | int64 | 7855 | 6649 | 878000000  | NO |
|  original_balance | int64 | 7855 | 4241 | 878000000  | NO |
|  orig_int_rate | float64 | 7855 | 581 | 0.0333  | NO |
|  orig_dscr | float64 | 7855 | 686 | 2.27  | NO |
|  orig_ltv | float64 | 7855 | 569 | 0.524  | NO |
|  principal_paydown | int64 | 7855 | 5838 | 0  | YES (beg. - cur. bal.) |
|  orig_occ_rate | float64 | 7855 | 788 | 0.977  | NO |
|  most_rct_occ_rate | float64 | 7855 | 80 | 0.97  | NO |
|  occ_rate_delta | float64 | 7855 | 1210 | -0.007  | YES (cur. - orig.) |
|  orig_noi | float64 | 7855 | 6249 | 68071000.0  | NO |
|  most_rct_noi | int64 | 7855 | 5596 | 61830621  | NO |
|  noi_delta | float64 | 7855 | 7152 | -6240379.0  | YES (cur. - orig.) |
|  most_rct_value | int64 | 7855 | 2323 | 1675000000  | NO |
|  orig_value | float64 | 7855 | 7286 | 1675572519.08  | YES (orig. bal. / orig. ltv) |
|  value_delta | float64 | 7855 | 6787 | -572519.083969  | YES (cur. val. - orig. val.) |
|  most_rct_ltv | float64 | 7855 | 6088 | 0.524179104478  | YES (cur. bal. / cur. val.) |
|  ltv_delta | float64 | 7855 | 6123 | 0.000179104477612  | YES (cur. ltv. - org. ltv) |
|  most_rct_debt_serv | float64 | 7855 | 5603 | 22334125.0  | NO |
|  orig_debt_serv | float64 | 7855 | 6359 | 29987224.6696  | YES (orig. noi / orig. dscr) |
|  debt_serv_delta | float64 | 7855 | 7235 | -7653099.6696  | YES (cur. ds - orig. ds) |
|  most_rct_dscr | float64 | 7855 | 1682 | 2.74  | NO |
|  dscr_delta | float64 | 7855 | 2003 | 0.47  | YES (cur. dscr - orig. dscr) |
|  state_AK | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_AL | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_AR | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_AZ | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_CA | float64 | 7855 | 2 | 1.0  | state -> get_dummies |
|  state_CO | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_CT | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_DC | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_DE | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_FL | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_GA | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_HI | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_IA | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_ID | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_IL | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_IN | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_KS | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_KY | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_LA | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_MA | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_MD | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_ME | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_MI | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_MN | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_MO | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_MS | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_MT | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_NC | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_ND | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_NE | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_NH | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_NJ | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_NM | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_NV | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_NY | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_OH | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_OK | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_OR | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_PA | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_RI | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_SC | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_SD | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_TN | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_TX | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_UT | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_VA | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_WA | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_WI | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_WV | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  state_WY | float64 | 7855 | 2 | 0.0  | state -> get_dummies |
|  ss_arbor_commercial_mortgage,_llc | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_berkadia | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_berkeley_point_capital,_llc | float64 | 7855 | 2 | 1.0  | special_servicer -> get_dummies |
|  ss_c-iii_asset_management_llc | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_cwcapital | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_freddie_mac | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_gemsa | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_greystone_servicing_corporation,_inc. | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_keybank | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_midland_loan_services | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_pacific_life_insurance_company | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_sabal_financial_group,_l.p. | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_situs_holdings,_llc | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_torchlight | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_trimont | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_wells_fargo_bank | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |
|  ss_wells_fargo_bank_ | float64 | 7855 | 2 | 0.0  | special_servicer -> get_dummies |

** * label of "1.0" derived from:
  * dlq_status_text (status as mapped below** - 200, 300, 350, 450 were considered
    problem loans),
  * no_time_dlqlife (if loan had ever been deliquent counted as problem loan), and
  *  date_added_to_servicer_watchlist_ (if servicer had been put on watch, considered problem loan)

**Mapping functions:**  
 Map df_mspd dlq_status_text column to corresponding values for label determination:  

    100 = Current or less than 60 day delinquent  
    200 = 60 or more days delinquent  
    300 = Foreclosure  
    350 = Problem Loans (no_time_dlqlife > 0 or date_added_to_servicer_watchlist_ Non-null)  
    450 = Real estate owned  
    500 = Closed
** * Any values with a 500 were reverted to the highest previous value

## Analysis

Based on evaluation of loans foreclosed in the Freddie Mac portfolio, the average percentage of outstanding balance that is recovered following disposition is 0.6930486709.
