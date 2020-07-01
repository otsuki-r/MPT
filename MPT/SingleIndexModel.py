import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

class SingleIndexModel():
    """
    Class combining stocks with a benchmark index and risk-free return.

    Contains methods to computes the alpha and beta of stocks, against the benchmark,
    using their daily returns and a risk-free return.
    """
    default_figsize = (14, 8)
    
    def regressSecurityCharacteristicLine(self):
        """
        SCL is regression of excess security return against excess market return.
        We use a benchmark index as a proxy for market portfolio
        SCL: R_{i, t} - r_f = alpha_i + beta_i (R_{M, t} - r_f) + noise

        Returns
        -------
        statsmodels.regression.linear_model.RegressionResults
            Statsmodels result of the linear regression.
        -------
        """
        y = self.daily_returns - self.risk_free_return
        x = self.daily_benchmark_returns - self.risk_free_return

        # Add a constant column to allow for intercept (= alpha)
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        
        return model.fit()
    
    def __init__(self, daily_returns_df, daily_benchmark_returns, risk_free_return):
        self.daily_returns = daily_returns_df.fillna(0)
        self.daily_benchmark_returns = daily_benchmark_returns.fillna(0)
        self.risk_free_return = risk_free_return
        self.num_stocks = len(daily_returns_df.columns)
        self.stock_names = daily_returns_df.columns
        
        regression_model = self.regressSecurityCharacteristicLine()
        regression_coefficients = regression_model.params
        self.alphas = regression_coefficients.values[0]
        self.betas = regression_coefficients.values[1]
        self.residuals = regression_model.resid
        
    def plotSecurityCharacteristicLines(self, fig = None, ax = None):
        """
        Produces stacked plots of the SCL of each stock in self.daily_returns
        (one for each). Additionally, for each stock, plots a histogram of the
        residuals of the fit on a seperate axis.

        Returns
        -------
        mpt.Figure
            New or modified figure.
        mpt.Axes
            New or modified axes.
        -------
        """
        if ax is None:
            fig, ax = plt.subplots(figsize = (14, 5 * self.num_stocks))
        
        # Create a set of axes for each stock
        grid_aspect_ratio = (self.num_stocks * 5, 8)
        scl_axes = [plt.subplot2grid(grid_aspect_ratio,
                                     (5 * j, 0),
                                     rowspan = 4,
                                     colspan = 6,
                                    ) for j in range(self.num_stocks)]
        
        residuals_axes = [plt.subplot2grid(grid_aspect_ratio,
                                           (5 * j, 6),
                                           rowspan = 4,
                                           colspan = 2,
                                          ) for j in range(self.num_stocks)]
        
        for j in range(self.num_stocks):
            # for each stock
            this_stock_name = self.stock_names[j]
            this_alpha = self.alphas[j]
            this_beta = self.betas[j]
            this_residual = self.residuals[this_stock_name]
            this_scl_axis = scl_axes[j]
            this_residual_axis = residuals_axes[j]

            # Plot scatter plot of excess returns v.s. benchmark excess returns
            y = self.daily_returns[this_stock_name] - self.risk_free_return
            x = self.daily_benchmark_returns - self.risk_free_return
            this_scl_axis.scatter(x, y)
            
            # Plot security characteristic line (regression line) on same plot
            predictor = np.linspace(min(x), max(x), 100)
            predicted = [this_alpha + this_beta * x for x in predictor]
            this_scl_axis.plot(predictor, predicted, 'k', linewidth = 2)
            
            # Plot 
            this_residual_axis.hist(this_residual, bins = 15)
            
            # Force symmetric axis of residual histogram
            this_residual_axis.set_xlim(- 1 * max(this_residual), max(this_residual))
            
            # Put y-axis of residuals on right hand side.
            this_residual_axis.yaxis.set_label_position("right")
            this_residual_axis.yaxis.tick_right()
            
            # Add information
            this_scl_axis.set_title(f'Security Characteristic Line for {this_stock_name}')
            this_scl_axis.set_xlabel('Excess benchmark returns')
            this_scl_axis.set_ylabel('Excess stock returns')
            
            this_residual_axis.set_title('Histogram of Residuals')
            this_residual_axis.set_xlabel('Residual')
            this_residual_axis.set_ylabel('Count')
            
        plt.show()
    
    def plotCompareSecurityCharacteristicLines(self, fig = None, ax = None):
        """
        Plots all the SCL on one plot, along with the benchmark for reference.

        Returns
        -------
        mpt.Figure
            New or modified figure.
        mpt.Axes
            New or modified axes.
        -------
        """

        if ax is None:
            fig, ax = plt.subplots(figsize = SingleIndexModel.default_figsize)
        

        # Get limits for plotting
        benchmark_excess_returns = self.daily_benchmark_returns - self.risk_free_return
        x = np.linspace(min(benchmark_excess_returns), max(benchmark_excess_returns), 100)


        for j in range(self.num_stocks):
            this_stock_name = self.stock_names[j]
            this_alpha = self.alphas[j]
            this_beta = self.betas[j]
            
            # Plot each stock performance
            ax.plot(x, this_alpha + this_beta * x , linewidth = 2, label = f'{this_stock_name}')

        # Plot reference benchmark v.s. benchmark
        ax.plot(x, x, 'k--', linewidth = 2, label = f'Benchmark')
        
        # Add information
        ax.set_title('Comparison of SCL against benchmark')
        ax.set_xlabel('Benchmark excess returns')
        ax.set_ylabel('Stock excess returns')
        ax.legend(loc = 'best')
        plt.show()
        return
