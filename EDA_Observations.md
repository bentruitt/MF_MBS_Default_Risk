# Observations of Data During EDA

## Column Data Types:

|**Column Name**      |**Non-Null** | **Type**     | **Convert** |
|                     |             |              |             |
|---------------------|:-----------:|:------------:|:-----------:|
|lnno                 | 338445      |  int64       | leave       |
|quarter              | 338445      |  object(str) | datetime    |
|mrtg_status          | 338445      |  int64       | get_dummies |
|amt_upb_endg         | 328999      |  float64     |  leave      |
|liq_dte              | 9160        |  object(str) | datetime    |
|liq_upb_amt          | 9164        |  float64     | leave       |
|dt_sold              | 73          |  object(str) | datetime    |
|cd_fxfltr            | 130063      |  object(str) | get_dummies |
|cnt_amtn_per         | 338428      |  float64     | leave       |
|cnt_blln_term        | 338428      |  float64     | leave       |
|cnt_io_per           | 338419      |  float64     | leave       |
|dt_io_end            | 98108       |  object(str) | datetime    |
|cnt_mrtg_term        | 338445      |  int64       | leave       |
|cnt_rsdntl_unit      | 338411      |  float64     | leave       |
|cnt_yld_maint        | 338445      |  int64       | leave       |
|code_int             | 338445      |  object      | get_dummies |
|rate_dcr             | 338290      |  float64     | leave       |
|rate_int             | 338416      |  float64     | leave       |  
|rate_ltv             | 338392      |  float64     | leave       |
|amt_upb_pch          | 338445      |  float64     | leave       |
|dt_fund              | 338445      |  object(str) | datetime    |
|dt_mty               | 338445      |  object(str) | datetime    |
|code_st              | 38445       |  object(str) | leave       |
|geographical_region  | 338048      |  object(str) | leave       |
|**id_link_grp**      | 66308       |  float64     | leave       |
|code_sr              | 19684       |  object(str) | get_dummies |
|reo_operating_expinc | 74          |  float64     | leave       |
|prefcl_fcl_expinc    | 74          |  float64     | leave       |
|selling_expinc       | 74          |  float64     | leave       |
|sales_price          | 73          |  float64     | leave       |


**'code_sr'** column contains code for type of senior care facility, if senior care facility. There are 19,684 senior care facilities, none of which have defaulted. Consider whether or not to include this data.
