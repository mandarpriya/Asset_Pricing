


library(tidyfinance)



# The q-factor model is an empirical asset pricing model proposed by Hou, Xue, and Zhang (2015).
# The model says that the expected return of an asset in excess of the riskfree rate is described by
# its sensitivities to the market factor, a size factor, an investment factor, and a return on equity
# factor.

#"RF" stands for the one-month Treasury bill rates, "MKT" the market excess returns, "ME" the size factor returns, "IA" the investment factor returns, "ROE" the return on equity factor returns, and "EG" the expected growth factor returns.


factors_q_monthly <- download_data(
  type = "factors_q5_monthly", 
  start_date = "1960-01-01"
  
)

factors_q_monthly 
