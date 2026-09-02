# Load Packages
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import xcdat as xc

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

## 
import scipy.signal
from scipy.signal import detrend

def detrend_index(index: xr.DataArray, label: str = "index") -> xr.DataArray:
    """
    Linearly detrend a 1D time series (e.g., climate index).

    Parameters
    ----------
    index : xr.DataArray
        Input time series (must have 'time' dimension)
    label : str
        Name prefix for output

    Returns
    -------
    xr.DataArray
        Detrended time series
    """

    detrended_index = xr.DataArray(
        detrend(index, type='linear'),
        coords=index.coords,
        dims=index.dims,
        name=f"{label}_detrended"
    )

    return detrended_index


### INDEX PLOTTING

## PLOTTING SINGLE INDEX [VALIDATION]

def single_index_plot(index_ref, 
                      index_name, 
                      index_ylabel):
    index_ref = index_ref
    index_time = index_ref.time
    
    # Main Figure
    plt.figure(figsize=(15, 5))

    index_ref.plot(color='black', lw=0.8, alpha=0.5)

    plt.fill_between(index_time, index_ref, 0, 
                     where=(index_ref >= 0), color='red', alpha=0.65)
    plt.fill_between(index_time, index_ref, 0, 
                     where=(index_ref < 0), color='blue', alpha=0.65)

    # Aesthetics
    plt.axhline(0, color='black', lw=1) # Zero line
    plt.title(index_name, fontsize=18, fontweight='bold')
    plt.ylabel(index_ylabel, fontsize=20)
    plt.xlabel('Time', fontsize=20)
    #plt.grid(True, alpha=0.3, ls='--')
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show()
    
    
def index_corr_plot(index1, index2, label1="Index 1", label2="Index 2", title="Index Comparison"):
    """
    Plots two timeseries, calculates their Pearson correlation, and displays it in the title.
    
    Parameters:
    index1, index2: xarray.DataArray or pandas.Series
    label1, label2: Strings for the legend
    title: Base string for the plot title
    """
    s1 = index1.to_series() if hasattr(index1, 'to_series') else index1
    s2 = index2.to_series() if hasattr(index2, 'to_series') else index2

    correlation = s1.corr(s2)

    fig, ax = plt.subplots(figsize=(15, 4))

    index1.plot(ax=ax, label=label1, linewidth=1.5, color='k')
    index2.plot(ax=ax, label=label2, linewidth=1.2, linestyle='dashed', color='red', alpha=0.8)

    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_title(f"{title} (Correlation: {correlation:.2f})", fontsize=14, weight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel("Index Value (K)")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
    )
    plt.tight_layout()
    plt.show()

    return correlation
    
    
def triple_index_plot(index1, index2, index3, 
                      label1="Index 1", label2="Index 2", label3="Index 3", 
                      title="Index Comparison"):
    """
    Plots three timeseries on a single plot for visual comparison.
    
    Parameters:
    index1, index2, index3: xarray.DataArray or pandas.Series
    label1, label2, label3: Strings for the legend
    title: String for the plot title
    """
    
    # 1. Create the Plot
    plt.figure(figsize=(20, 5))
    
    # 2. Plotting 
    # Using original objects to leverage xarray/pandas built-in plotting & formatting
    index1.plot(label=label1, linewidth=1.5, color='black')
    index2.plot(label=label2, linewidth=1.2, linestyle='dashed', color='red', alpha=0.8)
    index3.plot(label=label3, linewidth=1.2, linestyle='dotted', color='blue', alpha=0.8)
    
    # 3. Styling
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Time")
    plt.ylabel("Index Value (K)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(frameon=False, loc='best')
    plt.tight_layout()
    
    plt.show()

def plot_climate_indices(series_dict, title='Index Comparison', ylabel='Index Value', figsize=(18, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for label, (da, style) in series_dict.items():
        default_style = {'linewidth': 1}
        default_style.update(style)
        da.plot(ax=ax, label=label, **default_style)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.axhline(0, color='black', lw=0.8, ls='-', alpha=0.3)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(series_dict),
        frameon=False,
    )
    plt.tight_layout()
    plt.show()
    
    
def plot_climate_indices_w_psd(
    series_dict,
    title='Index Comparison',
    ylabel='Index Value',
    figsize=(20, 4),
    standardize=False,
    time_slice=None,
    fs=None,
    corr_method='pearson',
    show_running_mean=True,
    running_mean_window=12,
    fill_anomalies=False,
):
    fig, (ax_psd, ax_ts, ax_corr) = plt.subplots(
        1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 4, 1]}
    )

    # --- Pre-process series ---
    processed = {}
    for label, (da, style) in series_dict.items():
        s = da.to_series() if hasattr(da, 'to_series') else da.copy()
        if time_slice is not None:
            s = s.loc[time_slice]
        if standardize:
            s = (s - s.mean()) / s.std()
        processed[label] = (s, style)

    # --- Time series plot (middle) ---
    for label, (s, style) in processed.items():
        default_style = {'linewidth': 1}
        default_style.update(style)
        color = style.get('color', None)

        if fill_anomalies:
            ax_ts.fill_between(s.index, s, 0,
                               where=(s >= 0), color=color, alpha=0.3, linewidth=0)
            ax_ts.fill_between(s.index, s, 0,
                               where=(s < 0),  color=color, alpha=0.3, linewidth=0)

        ax_ts.plot(s.index, s.values, label=label, **default_style)

        if show_running_mean:
            rm = s.rolling(running_mean_window, center=True, min_periods=1).mean()
            ax_ts.plot(rm.index, rm.values, color=color,
                       linewidth=2.5, linestyle='-', alpha=0.5)

    ax_ts.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax_ts.set_ylabel(('Standardised ' if standardize else '') + ylabel, fontsize=12)
    ax_ts.set_xlabel('Time', fontsize=12)
    ax_ts.axhline(0, color='black', lw=0.8, ls='-', alpha=0.3)
    ax_ts.grid(True, linestyle=':', alpha=0.6)
    ax_ts.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(series_dict),
        frameon=False,
    )

    # --- PSD plot (left) ---
    all_periods = []
    all_psds    = []

    for label, (s, style) in processed.items():
        s_clean = s.dropna()

        if fs is not None:
            _fs = fs
        else:
            dt_days = pd.to_timedelta(np.diff(s_clean.index).mean()).days
            _fs = 365.25 / dt_days

        freqs, psd = scipy.signal.welch(
            s_clean.values, fs=_fs, nperseg=min(256, len(s_clean) // 2)
        )
        freqs, psd = freqs[1:], psd[1:]
        period = 1.0 / freqs
        all_periods.append(period)
        all_psds.append(psd)

        color = style.get('color', None)
        ax_psd.plot(period, psd, color=color,
                    linestyle=style.get('linestyle', '-'),
                    linewidth=style.get('linewidth', 1))

        # Red noise + 95% CI
        s_vals    = s_clean.values
        alpha_ar1 = np.corrcoef(s_vals[:-1], s_vals[1:])[0, 1]
        red_noise = (1 - alpha_ar1**2) / (
            1 - 2 * alpha_ar1 * np.cos(2 * np.pi * freqs / _fs) + alpha_ar1**2
        )
        red_noise *= psd.mean() / red_noise.mean()
        ci95 = red_noise * scipy.stats.chi2.ppf(0.95, df=2) / 2

        ax_psd.plot(period, red_noise, color=color, linestyle=':', linewidth=0.8, alpha=0.6)
        ax_psd.fill_between(period, red_noise, ci95, color=color, alpha=0.08)

        # Annotate dominant peak
        peak_idx    = np.argmax(psd)
        peak_period = period[peak_idx]
        peak_power  = psd[peak_idx]
        ax_psd.annotate(f'{peak_period:.1f}yr',
                        xy=(peak_period, peak_power),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color=color)

    ax_psd.set_xscale('log')
    ax_psd.set_yscale('log')

    # --- X axis: explicit period ticks ---
    min_period = max(0.5, min(p.min() for p in all_periods))
    max_period = min(50,  max(p.max() for p in all_periods))
    ax_psd.set_xlim(min_period, max_period)

    period_ticks = [0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
    visible_xticks = [p for p in period_ticks if min_period <= p <= max_period]
    ax_psd.set_xticks(visible_xticks)
    ax_psd.set_xticklabels([str(p) for p in visible_xticks], fontsize=7, rotation=45, ha='right')
    ax_psd.xaxis.set_minor_locator(plt.NullLocator())

    # --- Y axis: explicit power ticks ---
    all_psd_vals = np.concatenate(all_psds)
    psd_min = 10 ** np.floor(np.log10(np.nanmin(all_psd_vals)))
    psd_max = 10 ** np.ceil(np.log10(np.nanmax(all_psd_vals)))
    ax_psd.set_ylim(psd_min, psd_max)

    log_min = int(np.log10(psd_min))
    log_max = int(np.log10(psd_max))
    power_ticks = [10**i for i in range(log_min, log_max + 1)]
    ax_psd.set_yticks(power_ticks)
    ax_psd.set_yticklabels([f'$10^{{{i}}}$' for i in range(log_min, log_max + 1)], fontsize=7)
    ax_psd.yaxis.set_minor_locator(plt.NullLocator())

    ax_psd.set_xlabel('Period (years)', fontsize=12)
    ax_psd.set_ylabel('Power', fontsize=12)
    #ax_psd.set_title('PSD', fontsize=10, fontweight='bold', pad=10)
    ax_psd.grid(True, linestyle=':', alpha=0.6, which='major')

    # --- Triangle correlation matrix (right) ---
    labels      = list(processed.keys())
    n           = len(labels)
    series_list = [s for s, _ in processed.values()]

    corr_matrix = np.array([
        [s1.corr(s2, method=corr_method) for s2 in series_list]
        for s1 in series_list
    ])

    pval_matrix = np.ones((n, n))
    for i, s1 in enumerate(series_list):
        for j, s2 in enumerate(series_list):
            if i != j:
                aligned = pd.concat([s1, s2], axis=1).dropna()
                _, p = scipy.stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                pval_matrix[i, j] = p

    mask   = np.triu(np.ones((n, n), dtype=bool), k=1)
    masked = np.where(mask, np.nan, corr_matrix)

    im = ax_corr.imshow(masked, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')

    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                val        = corr_matrix[i, j]
                text_color = 'white' if abs(val) > 0.6 else 'black'
                sig_marker = '' if pval_matrix[i, j] < 0.05 else '*'
                ax_corr.text(j, i, f'{val:.2f}{sig_marker}',
                             ha='center', va='center',
                             fontsize=8, color=text_color, fontweight='bold')

    ax_corr.set_xticks(range(n))
    ax_corr.set_yticks(range(n))
    ax_corr.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax_corr.set_yticklabels(labels, fontsize=8)
    #ax_corr.set_title(f'Corr ({corr_method})', fontsize=10, fontweight='bold', pad=10)

    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04, label='r')

    plt.tight_layout()
    plt.show()
    return fig, (ax_psd, ax_ts, ax_corr)

