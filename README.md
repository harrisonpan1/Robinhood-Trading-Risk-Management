# Robinhood-Trading-Journal

This project is python scripts, which uses supported Robinhood API, [robin_stocks](https://github.com/jmfernandes/robin_stocks), and [Dash feature from Plotly](https://plotly.com/dash/open-source/) for 
creating visualized trading journal. You can varify and download to use it as personal trading journal for Robinhood and option risk management. 
<br><br>

> **Note: robin_stocks package was [not fully updated](https://stackoverflow.com/questions/79291035/robin-stocks-robinhood-authentication-stopped-working). Need to pull the repo manually instead of pip update. You can just pull from my forked version since i have made some updates in the robin_stocks source code** 
~~~
>> cd [PATH]/Robinhood-trading-journal/dashApp
>> git clone https://github.com/harrisonpan1/robin_stocks.git
>> cd robin_stocks
>> pip install .
~~~
<br>


The reason behind creating is that new age invester's trading habits in robinhood is very risky and rewarding and there is no budget tool or risk management tool, especially for option trading. 
<br><br>

> **Note: Python version=3.12.11.** 
<br>

The defalts endpoints to access robinhood api is in [login.py](https://github.com/harrisonpan1/Robinhood-Trading-Risk-Management/blob/main/dashApp/login.py), it is built based on robin_stock and has some helper functions to get trading info you want.  

The [trading_journal.py](https://github.com/harrisonpan1/Robinhood-Trading-Risk-Management/blob/main/dashApp/trading_journal.py) scripts have methods on accessing order history for stocks and options, current open postions for stock and options, greek risk managment tool for entire portfolio.
Its calculation are on FIFO basis as per [Robinhood's default cost basis](https://robinhood.com/us/en/support/articles/cost-basis/). 

Refer to [risk_position_manager.ipynb](https://github.com/harrisonpan1/Robinhood-Trading-Risk-Management/blob/main/dashApp/risk_position_manager.ipynb) for an example 
<br>




  
# Credits/Acknowledgement

Project: [Robinhood-Trading-Journal]（https://github.com/virajkothari7/Robinhood-Trading-Journal)
<br>