import numpy as np
import matplotlib.pyplot as plt
import scipy

class PortfolioContainer:
    """
    Class for curating different portfolios. Some plotting capabilities.
    """
    default_figsize = (14, 8)
    
    def __init__(self, stocks_df, stock_returns):
        self.stocks = stocks_df
        self.weights = []
        self.labels = []
        self.pf_numbers = []
        self.covariance_matrix = np.mat(self.stocks.cov())
        self.num_stocks = len(self.stocks.columns)
        self.stock_returns = stock_returns
        self.ef_stds = None
        self.ef_rets = None
    
    def addPortfolio(self, weights, label = None):
        """
        Adds a portfolio, represented by a list of weights, and other book-keeping
        devices to the container.

        Parameters
        ----------
        weights : list
                  Weights of the portfolio to be added

        label : str, optional
                Name of the portfolio.
        ----------
        """
        # Normalise the weights by their sum to ensure they sum to 1.
        sum_weights = np.sum(weights)
        normalised_weights = np.mat([w / sum_weights for w in weights])

        if label is None:
            label = f'Portfolio {len(self.weights)}'

        self.labels.append(label)
        self.pf_numbers.append(len(self.weights))
        self.weights.append(normalised_weights)
        return
    
    def removePortfolio(self, n):
        """
        Removes the specified portfolio from the list.

        Parameters
        ----------
        n : int
            Portfolio number, accessible by `listPortfolios`.
        ----------
        """
        index = self.pf_numbers.index(n)
        self.weights.pop(index)
        self.labels.pop(index)
        self.pf_numbers.pop(index)
        return
    
    def listPortfolios(self):
        """
        Lists the portfolios contained in this PortfolioContainer.
        """
        for i in range(len(self.weights)):
            print(f'{self.pf_numbers[i]}) {self.labels[i]}: {self.weights[i]}')
        return

    def getReturn(self, weights):
        """
        Computes the returns of the portfolio represented by `weights`.

        Parameters
        ----------
        weights : list
                  list of weights of stocks in the portfolio.
        ----------
        """
        weights = np.matrix(weights)
        return np.dot(self.stock_returns, weights.transpose()).item()
    
    def getVariance(self, weights):
        """
        Computes the variance of the portfolio represented by `weights`.

        Parameters
        ----------
        weights : list
                  list of weights of stocks in the portfolio.
        ----------
        """
        weights = np.matrix(weights)
        return (weights * self.covariance_matrix * weights.transpose()).item() * len(self.stocks)
    
    def getStandardDeviation(self, weights):
        """
        Computes the standard deviation of the portfolio represented by `weights`.

        Parameters
        ----------
        weights : list
                  list of weights of stocks in the portfolio.
        ----------
        """
        weights = np.matrix(weights)
        return self.getVariance(weights) ** 0.5
    
    def addRandomPortfolios(self, N):
        """
        Adds the specified number of portfolios with random weights to the container.

        Parameters
        ----------
        N : int
            Number of random portfolios to add.
        ----------
        """
        for _ in range(N):
            random_weights = np.random.dirichlet(np.ones(self.num_stocks)/ 10.)
            self.addPortfolio(random_weights)
        return
    
    def plotSampleRiskReturn(self, N, fig = None, ax = None):
        """
        Plots an MC simulation of the risk-return spectrum. Note that these
        MC portfolios are not added to the container permanently; they are removed
        after the plotting is finished.

        Parameters
        ----------
        N : int
            Number of MC simulations to run.
        fig : mpt.Figure, optional
            Figure instance to plot on.
        ax : mpt.Axes, optional
            Axes instance to plot on.
        ----------

        Returns
        -------
        mpt.Figure
            New or modified figure.
        mpt.Axes
            New or modified axes.
        -------
        """
        if ax is None:
            fig, ax = plt.subplots(figsize = PortfolioContainer.default_figsize)
            
        temp_pf_numbers = self.pf_numbers.copy()
        temp_weights = self.weights.copy()
        temp_labels = self.labels.copy()
        self.pf_numbers = []
        self.weights = []
        self.labels = []
        self.addRandomPortfolios(N)
        
        pf_stds = [self.getStandardDeviation(w) for w in self.weights]
        pf_rets = [self.getReturn(w) for w in self.weights]
        
        ax.scatter(pf_stds, pf_rets, s = 10)
        self.pf_number = self.pf_numbers
        self.weights = temp_weights
        self.labels = temp_labels

        #ax.scatter(portfolio_stds, portfolio_rets, s = 4)
        ax.set_title('Risk-Return Profile')
        ax.set_xlabel('Standard deviation of portfolios')
        ax.set_ylabel('Return of portfolios')
        return fig, ax

    def plotRiskReturn(self, fig = None, ax = None):
        """
        Plot the risk-return profile of all portfolios in the container.

        Parameters
        ----------
        fig : mpt.Figure, optional
            Figure instance to plot on.
        ax : mpt.Axes, optional
            Axes instance to plot on.
        ----------

        Returns
        -------
        mpt.Figure
            New or modified figure.
        mpt.Axes
            New or modified axes.
        -------
        """
        if ax is None:
            fig, ax = plt.subplots(figsize = PortfolioContainer.default_figsize)
        
        
        pf_stds = [self.getStandardDeviation(w) for w in self.weights]
        pf_rets = [self.getReturn(w) for w in self.weights]
        
        for i in range(len(self.weights)):
            ax.scatter(x = pf_stds[i],
                       y = pf_rets[i],
                       marker = 'x', 
                       s = 100,
                       label = self.labels[i],
                      )

        #ax.scatter(portfolio_stds, portfolio_rets, s = 4)
        ax.set_title('Risk-Return Profile')
        ax.set_xlabel('Standard deviation of portfolios')
        ax.set_ylabel('Return of portfolios')
        ax.legend(loc = 'best')
        
        return fig, ax
    
    
    def plotNormalisedPortfolios(self, fig = None, ax = None):
        """
        Plot the normalised returns of each portfolio in the container

        Parameters
        ----------
        fig : mpt.Figure, optional
            Figure instance to plot on.
        ax : mpt.Axes, optional
            Axes instance to plot on.
        ----------

        Returns
        -------
        mpt.Figure
            New or modified figure.
        mpt.Axes
            New or modified axes.
        -------
        """

        if ax is None:
            fig, ax = plt.subplots(figsize = PortfolioContainer.default_figsize)
        for i in range(len(self.weights)):
            pf_value = self.stocks.multiply(np.squeeze(np.asarray(self.weights[i])), axis = 1).sum(axis = 1)
            ax.plot(self.stocks.index, np.cumprod(1 + pf_value.values),
                    linewidth = 2,
                    label = f'{self.labels[i]}')
        ax.set_title('Normalised Change in Value of portfolios')
        ax.set_ylabel('Relative Change')
        ax.legend(loc = 'best')
        
        return fig, ax
        
    def plotEfficientFrontier(self, fig = None, ax = None):
        """
        Plots the efficient frontier of the set of stocks in the `PortfolioContainer`.

        Parameters
        ----------
        fig : mpt.Figure, optional
            Figure instance to plot on.
        ax : mpt.Axes, optional
            Axes instance to plot on.
        ----------

        Returns
        -------
        mpt.Figure
            New or modified figure.
        mpt.Axes
            New or modified axes.
        -------
        """
        def initialiseEfficientFrontierLimits():
            """
            Runs an MC simulation of `N` random portfolios to get a feel for the range
            of risk values that we need to maximise over.

            Note that the randomly generated portfolios are not added permanently to the
            container; they are removed after we are done.

            Returns
            -------
            float
                Smallest value of the standard deviation observed.
            float
                Largest value of the standard deviation observed.
            -------
            """
            N = 2500
            # Find range of standard deviations for plotting efficient frontier
            # Temporarily adds N random portfolios which are discarded at the end

            temp_pf_numbers = self.pf_numbers.copy()
            temp_weights = self.weights.copy()
            temp_labels = self.labels.copy()

            self.addRandomPortfolios(N)
            stds = [self.getStandardDeviation(w) for w in self.weights]
            rets = [self.getReturn(w) for w in self.weights]

            self.pf_number = self.pf_numbers
            self.weights = temp_weights
            self.labels = temp_labels
            return min(stds), stds[rets.index(max(rets))]

        def maximiseRetAndMinimiseStd(max_std):
            """
            Computes the weights that maximises the returns at the given level of risk.

            Parameters
            ----------
            max_std : float
                      Fixed level of risk to maximise the return at.
            ----------

            Returns
            -------
            np.mat
                Weights that maximises the returns at the specified level of risk.
            -------
            """
            # Two constraints:
            # 1) weights must sum to 1
            constraint1 = {'type':'eq',
                           'fun': lambda w: sum(w) - 1.}
    
            # 2) std of the desired portfolio must be below the specified `max_std`
            constraint2 = {'type': 'ineq',
                           'fun': lambda w: - 1. * (self.getStandardDeviation(np.mat(w)) - max_std)}
        

            # We now compute the weights that optimise these constraints.

            # Rather than `scipy.optimize.minimize`, which finds local minima, we use
            # `scipy.optimize.basinhopping` which is designed to find global minima.
            # It works by calling `minimize` repeatedly after taking random steps in the
            # parameter space to try and climb out of any local minima.

            # Apply `bounds` such that all weights to lie between 0 and 1.
            minimizer_kwargs = dict(bounds = [(0, 1)] * self.num_stocks,
                                    constraints = [constraint1, constraint2]
                                    )

            # maximising returns is equivalent to minimising negative returns.
            # Take initial guess `x0` for weights to be uniform.
            optimal_weights = scipy.optimize.basinhopping(lambda w: - 1 * self.getReturn(w),
                                                          x0 = [1. / self.num_stocks] * self.num_stocks,
                                                          minimizer_kwargs = minimizer_kwargs
                                                         )
            return np.mat(optimal_weights.x)
    
        def constructEfficientFrontier(ef_lower_lim, ef_upper_lim):
            """
            Construct the efficient frontier by optimising weights between limits.
            We optimise the returns at `num_samples` equally spaced points between the
            specified arguments.

            Bounds expected to have been obtained from `initialiseEfficientFrontierLimits`.

            Parameters
            ----------
            ef_lower_lim : float
                           Lower limit of range for which we compute the efficient frontier.
            ef_upper_lim : float
                           Upper limit of range for which we compute the efficient frontier.
            ----------

            Returns
            -------
            list
                Standard deviations of portfolios on the efficient frontier.
            list
                Returns of portfolios on the efficient frontier.
            -------
            """
            num_samples = 25
            ef_std= np.linspace(ef_lower_lim, ef_upper_lim, num = num_samples).tolist()
            ef_weights = [maximiseRetAndMinimiseStd(std) for std in ef_std]
            ef_returns = [self.getReturn(w) for w in ef_weights]

            return ef_std, ef_returns
        
        if ax is None:
            fig, ax = plt.subplots(figsize = PortfolioContainer.default_figsize)

        if self.ef_stds is None:
            ef_lower_lim, ef_upper_lim = initialiseEfficientFrontierLimits()
            self.ef_stds, self.ef_rets = constructEfficientFrontier(ef_lower_lim, ef_upper_lim)
        
        # Plot the efficient frontier
        ax.plot(self.ef_stds, self.ef_rets, 'k', linewidth = 2, label = 'Efficient Frontier')

        # Add Information
        ax.legend(loc = 'best')
        return fig, ax
 
