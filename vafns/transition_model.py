import numpy as np
import torch
from vafns.dvbf import TransitionModel


def calc_forward_rate(price_t1, price_t2, time_difference):
    """
    Calculates the forward rate between bonds with different
    maturity dates. Assumes continuously compounded interest
    rates.
    Parameters
    ----------
    price_t1: array_like
        The price of a bond with t1 number of years remaining.
    price_t2: array_like
        The price of a bond with t2 number of years remaining.
    time_difference: float
        The difference between t2 and t1.
        t2 is greater than t1.
    Returns
    -------
    forward_rate: array_like
        The implied forward rate from price_t1 and price_t2
    """

    forward_rate = (np.log(price_t1)-np.log(price_t2)) / time_difference

    return forward_rate


def create_wiener(time_steps, paths):
    """
    Creates random variables using a standard normal distribution.
    Parameters
    ----------
    time_steps: int
        Total number of time steps.
    paths: int
        Total number of paths.
    Returns
    -------
    dw: array_like
        Array of standard normal random variables
    """

    # Set seed:
    torch.random.seed(123)

    # Get standard normal random variables:
    dw = torch.randn(time_steps, paths)

    return dw


class InterestRates(object):
    """
    Class for risk-free interest rate dynamic construction.
    Parameters
    ----------
    initial_rate: float
        Initial annual interest rate.
    terminal_period: int
        Value for the terminal time period.
        Must be greater than current_period.
    current_period: int
        Value for the current time period.
        Must be non-negative and less than terminal_period.
        Defaults to 0.
    """

    def __init__(self, initial_rate, terminal_period, current_period=0):
        self.initial_rate = initial_rate
        self.terminal_period = terminal_period
        self.current_period = current_period

        # Check time periods:
        self.check_time_periods()

    def check_time_periods(self):
        """
        Check to see if terminal_period is greater than current_period:
        """

        if self.current_period > self.terminal_period:
            raise ValueError("terminal_period needs to be greater "
                             "than or equal to current_period")

    @staticmethod
    def calc_affine_term_structure(r_t, a_t, b_t):
        """
        Calculates the price of a zero-coupon bond using an affine
        term structure.
        Parameters
        ----------
        r_t: float
            Annual interest rate at time t.
        a_t: float
            Coefficient for an affine term structure model.
        b_t: float
            Coefficient for an affine term structure model.
        Returns
        -------
        price: float
            Price of a zero-coupon bond.
        """

        price = a_t * np.exp(-b_t*r_t)

        return price

    def setup_paths(self, dt, paths):
        """
        Sets up initial array for interest rate paths used for Monte
        Carlo simulations.
        Parameters
        ----------
        dt: float
            The time interval.
        paths: int
            Total number of paths.
        Returns
        -------
        r_array: array_like
            Initial array of interest rate paths.
        """

        # Calculate the total number of time steps:
        time_steps = int((self.terminal_period -
                          self.current_period)/dt) + 1

        # Create initial array of interest rates (time_steps x paths):
        r_array = np.zeros([time_steps, paths])

        # Initialize first row as self.initial_rate:
        r_array[0, :] = self.initial_rate

        return r_array

    def calc_monte_carlo_price(self, dt, paths):
        """
        Calculates the price of a zero-coupon bond at each dt time
        step using Monte Carlo simulations. The last index of the
        returned array corresponds to the price of a zero-coupon bond
        with maturity self.terminal_period.
        Parameters
        ----------
        dt: float
            The time interval.
        paths: int
            Total number of paths.
        Returns
        -------
        price: array_like
            Price of a zero-coupon bond.
        """

        # Create Monte-Carlo paths:
        mc = self.create_paths(dt, paths=paths)[1:]

        # Get the cumulative sum of each path (i.e. compute the
        # integral):
        mc_sum = np.cumsum(mc*dt, axis=0)

        # Find the price of a zero-coupon bond for each time step:
        price = np.mean(np.exp(-mc_sum), axis=1)

        return price

    def calc_zero_rate(self, r_t=None):
        """
        Calculates the spot rate of a zero-coupon bond.
        Parameters
        ----------
        r_t: float
            Annual interest rate at time t.
            Default is None.
        Returns
        -------
        spot_rate: float
            Spot rate at time t.
        """

        # Check if r_t is None:
        if r_t is None:
            r_t = self.initial_rate

        # Calculate the price of a zero-coupon bond:
        price = self.calc_zero_coupon_price(r_t)

        # Calculate time difference:
        time_diff = self.terminal_period - self.current_period

        # Calculate spot rate:
        spot_rate = -(1.0/time_diff) * np.log(price)

        return spot_rate

class VasicekRatesTransitionModel(InterestRates, TransitionModel):
    """
    Class for Vasicek risk-free interest rate short-rate model.
    Parameters
    ----------
    initial_rate: float
        Initial annual interest rate.
    theta: float
        Speed of mean reversion.
        Must be non-negative.
    mu: float
        Mean of risk-free interest rate.
        Must be non-negative.
    sigma: float
        Volatility of risk-free interest rate.
        Must be non-negative.
    terminal_period: int
        Value for the terminal time period.
        Must be greater than current_period.
    current_period: int
        Value for the current time period.
        Must be non-negative and less than terminal_period.
        Defaults to 0.
    """

    def __init__(self, initial_rate, theta, mu, sigma,
                 terminal_period, current_period=0):
        InterestRates.__init__(self, initial_rate, terminal_period,
                               current_period)
        TransitionModel.__init__(self, latent_dim=latent_dim, action_dim=action_dim, noise_dim=noise_dim)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def calc_zero_coupon_price(self, r_t):
        """
        Calculates the price of a zero-coupon bond using an affine
        term structure.
        Parameters
        ----------
        r_t: float
            Annual interest rate at time t.
        Returns
        -------
        price: float
            Price of a zero-coupon bond.
        """

        # Calculate the value of B(self.current_period,
        # self.terminal_period):
        b_t = ((1.0 - np.exp(-self.theta*(self.terminal_period -
                                          self.current_period)))
               / self.theta)

        # Calculate the value of A(self.current_period,
        # self.terminal_period):
        first_term = ((b_t - self.terminal_period +
                       self.current_period)*(
                              (self.theta**2*self.mu) -
                              0.5*self.sigma**2))/(self.theta**2)
        second_term = (self.sigma**2) * (b_t**2) / (4.0*self.theta)

        a_t = np.exp(first_term-second_term)

        # Calculate the price:
        price = self.calc_affine_term_structure(r_t, a_t, b_t)

        return price

    def create_paths(self, dt, paths=1):
        """
        Creates interest rate path using the underlying short-rate
        model.
        Parameters
        ----------
        dt: float
            The time interval.
        paths: int
            Number of paths to create.
            Default value is 1.
        Returns
        -------
        r_array: array_like
            Array of interest rate paths following the underlying
            short-rate model
        """

        # Create initial array of interest rates:
        r_array = self.setup_paths(dt, paths)

        # Get time steps:
        time_steps = r_array.shape[0]

        # Create standard normal random variables:
        dw = create_wiener(time_steps, paths)

        # Loop through to create interest rate paths:
        for t in range(1, time_steps):
            r_array[t, :] = r_array[t - 1, :] + \
                            self.theta*(self.mu -
                                        r_array[t - 1, :])*dt + \
                            self.sigma*np.sqrt(dt)*dw[t, :]

        return r_array

    def forward(self, latent, noise):
        pass



#class VasicekTRansitionModel inheriting the Transition model, init defining the vasicek rates object, used in the forward method
#latent and noise--wiener
