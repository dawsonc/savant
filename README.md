Welcome to Savant
=================

Savant is an artificial neural network system for analyzing and predicting changes in the stock market, targeted at predicting shifts in the S&P 500 (as modeled by the SPY exchange-traded-fund).

30,000-foot overview
--------------------

Objective: Given historical market data, predict future changes in the SPY ETF (which models the S&P 500 index)

Indicators:
    + SPY (last 30 days)
        * An ETF that models the S&P 500 index
    + USO (last 30 days)
        * An ETF that tracks the performance of the price of crude oil
    + SHY (last 30 days)
        * An ETF that tracks an index of U.S. Treasury bonds with maturities between one and three years

*Note: We have decided to use ETFs because, as they are traded like securities, historical pricing data is readily available for ETFs, simplifying the data-acquisition component of the system pipeline*

Machine Learning Technique: Artificial Neural Network

Technology Stack: Scipy, Numpy, Pandas, & matplotlib (iPython used for development)

Modules
-------

+ `spy.py` contains the code for retrieving historical price data from the Yahoo Finance API.
+ `tailor.py` contains the code for extracting relevant data from the data retrieved by `retriever.py` & constructing feature vectors and matrices