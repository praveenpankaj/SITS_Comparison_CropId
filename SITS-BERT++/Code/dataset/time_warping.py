import numpy as np
from scipy.interpolate import interp1d

def time_warping(ts, sigma=0.2):
    """
    Apply time warping to a time series.
    
    :param ts: numpy array of shape (seq_len, num_features)
    :param sigma: standard deviation of the warping effect
    :return: time-warped time series
    """
    seq_len, num_features = ts.shape
    time_steps = np.arange(seq_len)

    # Generate random warping points and apply Gaussian noise
    random_points = np.linspace(0, seq_len - 1, num=4)
    random_shifts = np.random.normal(loc=0, scale=sigma, size=4)
    random_shifts[0] = random_shifts[-1] = 0  # No warping at the start and end points

    # Create interpolation function
    interp_func = interp1d(random_points + random_shifts, random_points, kind='linear')

    # Generate new time steps by applying the interpolation function
    new_time_steps = interp_func(time_steps)

    # Interpolate original time series to the new time steps
    warped_ts = np.zeros_like(ts)
    for i in range(num_features):
        warped_ts[:, i] = np.interp(new_time_steps, time_steps, ts[:, i])

    return warped_ts