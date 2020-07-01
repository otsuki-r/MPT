from scipy import stats

class ReturnsGetter:
    """
    Class to get returns from list of DataFrames of daily returns
    Takes additional argument to specify how returns should be computed.
    """
    def getAnnualisedReturns(self):
        """
        Computes the geometric mean of the annualised returns of a set of stocks.

        Returns
        -------
        float
            Geometric mean of Annualised returns computed.
        -------

        """
        # Re-sample returns to be yearly, aggregating by cumulative product
        yearly = self.historic_daily_returns.resample('Y').agg(lambda x: (1 + x).prod() - 1)

        # return the geometric mean of these annualised returns
        return ((1 + yearly).prod()**(1. / len(yearly)) - 1).tolist()
    
    def __init__(self, historic_daily_returns, method = 'annualised'):
            
        # Any missing data should be filled with zeros (no daily changes)
        self.historic_daily_returns = historic_daily_returns.fillna(0)
        get_returns = {'annualised': self.getAnnualisedReturns}
        
        self.get_returns = get_returns[method]()
