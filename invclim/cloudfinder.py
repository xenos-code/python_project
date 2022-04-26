"""Tool for finding cloud layers in a dataframe containing a single atmospheric sounding.

Requirements: 'adjusted_relative_humidity' and 'height' be in the dataframe already. Find some
data to back up a choice for height threshold. I set it at 75, but that partially drawn from a hat.
Nice but not necessary: integrate metpy units throughout.
"""
import numpy as np
from scipy.interpoalte import interp1d
import core

def cloud_finder(data):
    """Implementation of cloud detection algorithm from Zhang et al. 2013)."""
    
    def min_rh(z):
        """minimum relative humidity threshold for radiosonde cloud detection
        based on Zhang et al 2013."""
        xp = np.array([0,2,6,12])*1000 # height in meters
        fp = np.array([92,90,88,75])
        return interp1d(xp, fp, bounds_error=False, fill_value=75)(z)

    def max_rh(z):
        """maximum within-layer relative humidity threshold to classify layer as 
        cloud based on Zhang et al 2013."""
        xp = np.array([0,2,6,12])*1000 # height in meters
        fp = np.array([95,93,90,80])
        return interp1d(xp, fp, bounds_error=False, fill_value=75)(z)

    def int_rh(z):
        """minimum relative humidity threshold for merged interstitial layers 
        based on Zhang et al 2013. Discontinuity at 2 km in paper removed."""
        xp = np.array([0,2,6,12])*1000 # height in meters
        fp = np.array([84,81,78,70])
        return interp1d(xp, fp, bounds_error=False, fill_value=70)(z)

    def init_sign_vector_cloud(rh, z, height_thresh=0):
        """Flags potential cloud layers. rh is relative humidity and
        z is height in meters."""
        return (rh >  min_rh(z)) & (z >= z.min() + height_thresh)

    return init_sign_vector_cloud(data['adjusted_relative_humidity'], data['z'], height_thresh=75)