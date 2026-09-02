import xarray as xr
import xesmf as xe
import numpy as np
from scipy import signal

def wrap_lon_xr(obj, lon_name='lon'):
    """
    Converts xarray object longitude from 0-360 to -180-180.
    
    Parameters:
    obj (xr.Dataset or xr.DataArray): The input xarray object.
    lon_name (str): The name of the longitude coordinate (default 'lon').
    """
    # 1. Capture attributes to prevent losing 'units', etc.
    lon_attrs = obj[lon_name].attrs
    
    # 2. Apply the transformation
    # Logic: (lon + 180) % 360 - 180
    obj = obj.assign_coords({
        lon_name: (((obj[lon_name] + 180) % 360) - 180)
    })
    
    # 3. Sort by longitude (Crucial for plotting and slicing)
    obj = obj.sortby(lon_name)
    
    # 4. Restore attributes
    obj[lon_name].attrs = lon_attrs
    
    return obj

def regridding(var_in, var_out):
    var_regridder =xe.Regridder(var_in, var_out, 
                                method='bilinear', periodic=True)
    var_out = var_regridder(var_in, keep_attrs=True)
    
    return var_out

def compute_seasonal_anomalies(ds: xr.Dataset, var: str) -> xr.Dataset:
    """
    Compute seasonal anomalies for a variable and add it to the dataset.

    Parameters:
        ds: xr.Dataset with a 'time' dimension
        var: variable name to compute anomalies for

    Returns:
        xr.Dataset with a new '{var}_anom' variable added
    """
    seasonal_means = ds[var].groupby("time.season").mean("time")
    ds[f"{var}_anom"] = ds[var].groupby("time.season") - seasonal_means
    ds = ds.drop_vars('season')
    return ds

def compute_pct_anomalies_from_groupby(grouped_da, min_threshold=0.1):
    """
    Computes percentage anomalies for a GroupBy object (e.g., grouped by 'season').
    
    Parameters:
    grouped_da (xr.core.groupby.DataArrayGroupBy): Data grouped by season.
    min_threshold (float): Minimum value to avoid division by zero/noise.
    
    Returns:
    xr.DataArray: Combined seasonal percentage anomalies.
    """
    
    def calculate_group_pct(group):
        # 1. Compute mean for the specific group (season)
        climatology = group.mean(dim='time')
        
        # 2. Compute absolute anomaly
        abs_anomaly = group - climatology
        
        # 3. Compute percentage anomaly with threshold masking
        return (abs_anomaly / climatology.where(climatology > min_threshold)) * 100

    # Apply the function to each group and combine back together
    return grouped_da.map(calculate_group_pct)

def standardize_by_month(da):
    return da.groupby('time.month').map(lambda x: (x - x.mean()) / x.std())

def detrend_da(da, dim='time', type='linear'):
    """
    Detrend an xarray DataArray along a given dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray to detrend.
    dim : str
        Dimension along which to detrend (default: 'time').
    type : str
        Type of detrending: 'linear', 'constant', or 'quadratic' (default: 'linear').

    Returns
    -------
    xr.DataArray
        Detrended DataArray with same shape, coords, and attrs as input.
        NaN slices are preserved (not filled).
    """
    # Convert time dim to numeric axis index
    axis = da.dims.index(dim)

    # Mask and fill NaNs before detrending (scipy can't handle NaNs)
    nan_mask = np.isnan(da.values)
    filled = np.where(nan_mask, 0.0, da.values)

    if type == 'quadratic':
        # Manual quadratic detrend via polyfit/polyval
        t = np.arange(da.sizes[dim])
        
        def _quad_detrend(arr_1d):
            valid = ~np.isnan(arr_1d)
            if valid.sum() < 3:
                return arr_1d * np.nan
            coeffs = np.polyfit(t[valid], arr_1d[valid], 2)
            trend = np.polyval(coeffs, t)
            return arr_1d - trend

        detrended = np.apply_along_axis(_quad_detrend, axis, da.values)
    else:
        # 'linear' or 'constant' — handled by scipy.signal.detrend
        detrended = signal.detrend(filled, axis=axis, type=type)

    # Re-apply original NaN mask
    detrended[nan_mask] = np.nan

    return xr.DataArray(
        detrended,
        coords=da.coords,
        dims=da.dims,
        attrs=da.attrs,
        name=da.name
    )