import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from .ReturnsGetter import ReturnsGetter

class StockPlotter:
    """
    Class for visualising stock performances based on historic data.
    """
    default_figsize = (14, 8)
    def __init__ (self, securities_full_df):
        # Everything is currently grouped by stock name. We rejig everything to be sorted by stock attribute
        # Stock attributes that we want
        self.high = securities_full_df['High']
        self.low = securities_full_df['Low']
        self.open = securities_full_df['Open']
        self.close = securities_full_df['Close']
        self.volume = securities_full_df['Volume']
        self.adj_close = securities_full_df['Adj Close']
        self.index = securities_full_df.index
        self.daily_changes = self.adj_close.pct_change()[1:]
        self.num_points = len(self.high)
        self.num_stocks = len(self.high.columns)
        self.stock_names = self.high.columns.to_list()
    
    def plotOHLCV(self):
        """
        `matplotlib` no longer supports OHLC candlestick plots. The replacemant `mplfinance` library
        is lacking and so we shall coerce matplotlib's in-built boxplot function to create the
        plots that we want.
        
        Usually, the boxplot function takes an array of data as input and automatically computes
        various stats (mean, median, q1, q3 etc.) and plots boxes based on them.
        
        However, we shall use the box to plot the open and close prices and the whiskers to plot
        the high and low prices. Note that we require one box per day.
        
        However, the default boxplot function does not allow us to customise where to plot the various
        attributes and so we shall use the box-plotter `Axes.bxp` function instead. This plots boxes
        from dictionaries that we can construct manually from the data.
        
        We also plot a single plot for trade volumes at the bottom.

        Returns
        -------
        mpt.Figure
            Figure instance.
        (list(mpt.Axes), list(mpt.Axes))
            Tuple of lists. The first element is a list of the candlestick plot axes whilst
            the second element is a list of the volume plot axes.
        -------
        """
        # Plot OHLC candlestick plots and trade volume
        fig = plt.figure(figsize = (14, self.num_stocks * 6))
        
        # Create one set of axes per stock.
        # After the first is created, add successive plots with `shareax = ax1`
        grid_aspect_ratio = (self.num_stocks * 5, 1)
        ax1 = plt.subplot2grid(grid_aspect_ratio, (0,0), rowspan = 4, colspan = 1, fig = fig)
        
        candlestick_axes = [plt.subplot2grid(grid_aspect_ratio,
                                             (5 * j, 0),
                                             rowspan = 4,
                                             colspan = 1,
                                             fig = fig,
                                             sharex = ax1)
                            for j in range(1, self.num_stocks)]
        candlestick_axes.insert(0, ax1)
        
        # Create a twinned set of axes for each OHLC axis to plot volume.
        volume_axes = [ax.twinx() for ax in candlestick_axes]
        


        # Create one OHLC candlestick plot per stock (separate plots).
        # Each box in each plot corresponds to one day's data.
        # We shall use `ax.bxp` to plot boxplots which requires the properties of 
        # each box to stored as a dictionary.
        # Each plot is then stored as a list of such dictionaries.
        for stock_num in range(self.num_stocks):
            # Iterate over stocks.
            this_candlestick_axis = candlestick_axes[stock_num]
            this_volume_axis = volume_axes[stock_num]
            this_stock_name = self.stock_names[stock_num]
            
            # List of colours of each box in this plot.
            this_boxplot_colors = []
            
            # Extract and store required information.
            this_boxplot = []
            for i in range(self.num_points):
                # Read today's open, close, high, low values and create a dictionary
                # of properties with which to create today's box.
                this_box = {}
                
                # `Axes.bxp` requires the following attributes, at a minimum, to construct boxes:
                # 1) 'med': median
                # 2) 'q1': first quartile
                # 3) 'q3': third quartile
                # 4) 'whislo': lower bound of whisker
                # 5) 'whishi': uppoer bound of whisker
                
                # We don't need the median line for an ohlc plot but we can't turn it off.
                # Set to mean of open and close (so it doesn't affect y-axis) and then hide later.
                this_box['med'] = (self.open.iloc[i][stock_num] + self.close.iloc[i][stock_num])/2
                this_box['q1'] = self.close.iloc[i][stock_num]
                this_box['q3'] = self.open.iloc[i][stock_num]
                this_box['whislo'] = self.low.iloc[i][stock_num]
                this_box['whishi'] = self.high.iloc[i][stock_num]
                
                # Add this box's properties to the this plot's dictionary
                this_boxplot.append(this_box)
            
                # Specify colour of box by whether the stock closed above or below the open price
                if self.close.iloc[i][stock_num] < self.open.iloc[i][stock_num]:
                    this_boxplot_colors.append('red')
                else:
                    this_boxplot_colors.append('green')
            #Finished extracting data.
            
            # Plotting
            """
            We pass the following options to `ax.bxp`:
                *) `widths = 0.9`: increase box widths to reduce whitespace between boxes.
                *) `medianprops = median_line_properties`: hide median line.
                *) `patch_artist = True`: tells `matplotlib` that the boxes are patches,
                    i.e. surfaces, rather than just paths. Required so we can colour
                    the boxes later.
                *) `showcaps = False`: hide the caps at the ends of the whiskers.
                *) `showfliers = False`: no outliers.
                *) `positions = range(self.num_points): tThe positions of the boxes
                    defaults to [1, 2, ..., n] but this produces an offset when plotting
                    the trade volume (probably because plotting from 0 causes the boxes to
                    underflow the axes. Here, we don't care and force-plot the boxes on
                    positions [0, 1, ..., n-1] instead.
            
            `ax.bxp` returns a dictionary of attributes.
            """
            median_line_properties = dict(linewidth=0)# hide plotting line
            plot = this_candlestick_axis.bxp(this_boxplot,
                                             widths = 0.9,
                                             medianprops = median_line_properties,
                                             patch_artist = True,
                                             showcaps = False,
                                             showfliers = False,
                                             positions = range(self.num_points)
                                            )
            
            # Colour each box with the appropriate colour.
            for boxes, color in zip(plot['boxes'], this_boxplot_colors):
                plt.setp(boxes, color=color)

            
            
            """
            Plot trade volume, on the same plot, on the second y-axis.
            Note that this is actually plotting the trade volumes against a
            discrete axis since we ignore missing dates (data from Mondays are plotted
            adjacent to data from Friday).
            """
            vol = self.volume[this_stock_name].tolist()
            this_volume_axis.fill_between(range(self.num_points), vol, alpha = 0.15, color = 'c')
            
            # Add information
            this_candlestick_axis.set_title(f'{this_stock_name} Stock Price')
            this_candlestick_axis.set_ylabel('Stock Price')
            this_volume_axis.set_ylabel('Trade Volume')

            # Fiddling with axis limits
            """
            Change ticker frequency of x-axis.
            We shall plot 8 major ticks on each plot. Get the dates to be used at
            the major ticks. We append a dummy date in major_tick_names to deal with
            the offset problem described above.
            """
            major_tick_frequency = self.num_points // 8
            major_tick_locator = ticker.MultipleLocator(major_tick_frequency)
            major_tick_names = [self.index.date[0]] + self.index.date[0::major_tick_frequency].tolist()
            major_tick_formatter = ticker.FixedFormatter(major_tick_names)
            this_candlestick_axis.xaxis.set_major_locator(major_tick_locator)
            this_candlestick_axis.xaxis.set_major_formatter(major_tick_formatter)            
            
            """
            Zero the trade volume axis
            """
            this_volume_axis.set_ylim(0, this_volume_axis.get_yticks()[-1])
            
            """
            Align grids of the two y-axes by scaling the ticks of the right-hand axis
            to fit grid lines of the left-hand axis.
            """
            lim1 = this_candlestick_axis.get_ylim()
            lim2 = this_volume_axis.get_ylim()
            scale_axis_points = lambda x : lim2[0]+(x - lim1[0]) / (lim1[1] - lim1[0]) * (lim2[1] - lim2[0])
            ticks = scale_axis_points(this_candlestick_axis.get_yticks())
            this_volume_axis.yaxis.set_major_locator(ticker.FixedLocator(ticks))
        plt.show()
        return fig, (candlestick_axes, volume_axes)
        
    def plotDailyReturns(self):
        """
        Plots a histogram of the distribution of daily returns and an
        associated KDE for each (note that the defualt `plt.kde` of `pandas`
        assumes a Gausssian kernel).

        We also plot a second plot underneath with a boxplot of the distribution.

        Returns
        -------
        mpt.Figure
            Figure instance.
        ((mpt.Axes, mpt.Axes), mpt.Axes)
            Tuple of axes. The first element is a tuple of the axes of the histogram + KDE plot.
            THe second element is the axes of the boxplot.
            the second element is a list of the volume plot axes.
        -------
        """

        spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 1])
        fig = plt.figure(figsize = (14, 8))
        ax1 = fig.add_subplot(spec[0])
        ax2 = ax1.twinx()
        ax3 = fig.add_subplot(spec[1], sharex = ax1)
        
        # Plot histogram of daily changes on left-hand axis of top plot
        # and KDE of histogram on right-hand axis of top plot
        # Boxplot on bottom plot.
        self.adj_close.pct_change()[1:].plot.hist(bins = 10, alpha = 0.15, ax = ax1)
        self.adj_close.pct_change()[1:].plot.kde(linewidth = 2, ax = ax2)
        self.adj_close.pct_change()[1:].boxplot(ax = ax3, vert = False)

        # Align grid of the two axes
        # Force both axes to start from 0 and then align gridlines of the two axes
        ax1.set_ylim(0, ax1.get_yticks()[-1])
        ax2.set_ylim(0, ax2.get_yticks()[-1])
        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
        
        # Hide the x-axis of the histogram
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Reduce space between histogram and boxplot
        plt.subplots_adjust(hspace = .01)

        # Add information
        ax1.set_title('Histogram of daily returns (left) and KDE (right)')
        ax1.set_ylabel('Frequency')
        ax2.set_ylabel('Kernel Density Estimation')
        ax3.set_xlabel('Daily returns')
        plt.show()
        return fig, ((ax1, ax2), ax3)
