# Analysis of Multifamily Mortgage Default Probabilities

## Project Introduction
The focus of this analysis is to approximate the probability of default for multifamily mortgages. I have worked in the area of multifamily lending for about 10 years. I have done everything from underwrite to originate to the sale of securities to large investment banks that would re-securitize the loans for consumers such as large institutional investors and hedge funds.

<img align="right" src="/img/thumbs_1600-Glenarm-002.jpg" alt="1600 Glenarm Place" width=40%>

The loan data that I will be using for this analysis represents the performance of Freddie Mac multifamily loans. Freddie Mac is the nickname used for the Federal Home Loan Mortgage Corporation (FHLMC). Freddie Mac works in partnership with a network of nationwide lenders to provide loans for both single-family and multifamily mortgages. A single-family property is defined as 1-4 units. A single-family mortgage is what you think of if you were to get a mortgage on your home. You may already have a Freddie Mac mortgage and not know it. A multifamily mortgage is what is provided for an apartment building or complex. A multifamily property is 5 units or more. An example of a multifamily property is 1600 Glenarm Place in Denver, CO. The focus of this analysis will be only multifamily mortgages.

## Freddie Mac

### U.S. Government Implicit Guarantee

<div style="float: right; width: 60%;">
  <img src="/img/investor-presentation.pdf.jpg" alt="Freddie Fed Bailout">
</div>

During the financial crisis in 2008, Freddie Mac received a bailout from the U.S. Government due to defaulting mortgages in their portfolio. Freddie Mac was given a loan for $71.3 Billion. Since this time, Freddie Mac has repaid $105.9 Billion in dividends. When the U.S. Government bailed-out Freddie Mac, the statement was made that Freddie Mac was "too big to fail" and had the implicit guarantee of the U.S. Government. Investors now began buying Freddie Mac mortgage securities as if they had no default risk. This reduced the interest rate required by investors to less than what would be required if the loan were made by a normal bank.

As of March 7th, 2017; if you look at the current interest rate required for a loan by Freddie Mac in comparison to a conventional / bank loan for a mortgage with a payment based on 30 years of amortization, term of 10 years, loan-to-value ratio of 80%, and a mortgage amount of $10,000,000, the rates would be:

* Freddie Mac Loan: 4.450%
* Conventional Loan: 4.711%

This does not look like a significant difference; however, let's assume you were the one investing the $10,000,000 to provide this loan and you did it based on Freddie Mac's default free status. Now the federal government announces that they will no longer provide their implicit guarantee. Your $10,000,000 investment now drops to a value of $9,809,682 a difference of $190,318 (or 1.90%). This is approximately the value associated with the default risk priced into the conventional loan. This is the value associated with the U.S. Government's implicit guarantee.

### Why Freddie Mac

<div style="float: right; width: 50%;">
  <img src="/img/freddie-mac-to-pay-treasury-4-5-billion-after-reporting-profit.jpg" alt="Freddie Fed Big Payment">
</div>

The political climate and public opinion has shifted. Freddie Mac appears to be standing on their own two feet and the implicit guarantee does not seem as necessary, in particular, now that Freddie Mac has paid back their loan. It is quite possible that the U.S. Government's implicit guarantee could be removed.

If the U.S. Government's implicit guarantee were to be removed it could significantly effect the value of Freddie Mac issued securities. This would be good for some and bad for others, but getting ahead of this by having a good understanding of the current default probabilities within the Freddie Mac loan portfolio and how to forecast the values that the securities would likely correct to is key in creating and profiting from the opportunities that will surely arise.

## Consumers of This Model

### Freddie Mac Security Investors
Two of the driving factors for investors to invest in Freddie Mac (and other GSE) securities are their ability to be leveraged, such as by hedge funds, and their ability to meet the Dodd-Frank Act requirement that banks and insurance companies hold higher percentages of risk free assets on their balance sheets.

The Dodd-Frank Act requirement is by far the largest domestic driver for the purchase of Freddie Mac securities. If a bank wants to increase their lending, they need to increase the amount of risk free assets on their balance sheets. If an insurance company wants to increase their liabilities, by issuing new policies, they need to hold more risk free assets. This aspect of the Dodd-Frank act is hendering the profitability of banks and insurance companies and they want it gone. They have been steadily lobbying for a change to this portion of the Dodd-Frank act and it may just be around the corner.

Hedge funds create leverage in their portfolios by leveraging their purchases. Leveraging is just another term for making purchases using a loan from somebody else. The purchases of risk free assets can be leveraged to a higher ratio of around 1-to-4 than purchases of assets that have a higher risk of default, such as, Commercial Mortgage Backed Security (CMBS) assets which may be leverage to around 1-to-3. If you look back at our Freddie Mac interest rate of 4.450%. A hedge fund may pay around 2.5% to borrow money, so after leverage they are making a return of just over 10% for the Freddie Mac loan investment and just under 8% for the conventional loan investment.

There are many likely to be affected by a removal of the U.S. Government's implicit guarantee, but a few that should be sure to pay particular attention to my results are insurance companies, banks, and hedge funds.

**Link to Presentation Slides:** [Default Risk In Multifamily Mortgages](https://docs.google.com/presentation/d/1AvFxeSGNUIpF76LP149ydiUKDZK0Xt6j5LOWwd6UNew/pub?start=true&loop=true&delayms=3000)

## Analysis

Based on evaluation of loans foreclosed in the Freddie Mac portfolio, the average percentage of outstanding balance that is recovered following disposition is 0.6930486709.
