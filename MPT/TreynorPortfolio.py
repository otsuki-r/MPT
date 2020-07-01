import numpy as np
import scipy.optimize

from .PortfolioContainer import PortfolioContainer
from .SingleIndexModel import SingleIndexModel

class TreynorPortfolio(PortfolioContainer):
    """
    Class to compute and plot the Treynor portfolio.
    """
    def getTreynorRatio(self, weights):
        """
        Computes the Treynor ratio. Like the Sharpe ratio, it rewards higher returns
        and penalises higher risk, though uses the Beta of the portfolio in place of
        it standard deviation to do so.


        Parameters
        ----------
        weights : np.matrix
            Weights of the portfolio. Weights must sum to 1.
        ----------


        Returns
        -------
        float
            Treynor ratio of `weights`
        -------
        """
        # Construct portfolio's returns and coerce to a dataframe
        this_portfolio = (self.historic_stock_returns * weights).sum(axis = 1).to_frame(name = 'pf1')

        # Construct a SIM, relative to the benchmark and risk-free return, to compute
        # this portfolio's beta
        sim = SingleIndexModel(this_portfolio,
                               self.historic_benchmark_returns,
                               self.risk_free_return
                               )

        return (self.getReturn(weights) - self.risk_free_return) / sim.betas
    
    def maximiseTreynorRatio(self):
        """
        Returns the weights of the portfolio that maximises the Treynor ratio
        under the supplied benchmark and risk-free return.

        Returns
        -------
        np.mat
            matrix of weights of the Treynor ratio
        -------
        """
        # Weights must sum to 1
        constraint1 = {'type': 'eq', 'fun': lambda x: sum(x) - 1.}
        minimised = scipy.optimize.minimize(lambda w: - 1. * self.getTreynorRatio(w),
                                            x0 = [1. / self.num_stocks] * self.num_stocks,
                                            bounds = [(0, 1)] * self.num_stocks,
                                            constraints = [constraint1],
                                            )

        if not minimised['success']:
            print('Failed to converge')
        return minimised['x']

    def __init__(self, historic_stock_returns, historic_benchmark_returns, stock_returns, risk_free_return):
        self.historic_stock_returns = historic_stock_returns.fillna(0)
        self.historic_benchmark_returns = historic_benchmark_returns.fillna(0)
        self.num_stocks = len(historic_stock_returns.columns)
        self.stock_returns = stock_returns
        self.risk_free_return = risk_free_return
        self.treynor_weights = self.maximiseTreynorRatio()
        return
