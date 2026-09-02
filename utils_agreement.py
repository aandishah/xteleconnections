import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_index_seasons(data_dict, seasons=["DJF", "MAM", "JJA", "SON"]):
    # Adjust number of subplots based on how many seasons you pass
    fig, axes = plt.subplots(1, len(seasons), figsize=(4.5 * len(seasons), 4), sharey=True)
    
    # Handle single season case (Matplotlib returns a single ax object, not a list)
    if len(seasons) == 1: axes = [axes]

    for ax, season in zip(axes, seasons):
        da = data_dict[season]
        
        # Year extraction logic
        t = da["time"]
        years = t.dt.year.values if hasattr(t, "dt") else np.array([pd.to_datetime(v).year for v in t.values])

        # Plotting
        y = da.values
        ax.bar(years, y, color=np.where(y > 0, "red", "blue"), width=0.9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(season, fontsize=12, fontweight="bold")
        
        # Ticks
        if len(years) > 0:
            step = max(1, int(np.ceil(len(np.unique(years)) / 6)))
            ax.set_xticks(np.unique(years)[::step])
            ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("ENSO Values")
    plt.tight_layout()
    plt.show()

def plot_enso_terciles_all_seasons(
    enso_seasonal,   # e.g. sst_seasonal_indices_LT.ENSO.groupby('time.season')
    top_dict,        # dict: season -> top_years
    thr_top_dict,    # dict: season -> thr_top
    bottom_dict,     # dict: season -> bottom_years
    thr_bot_dict,    # dict: season -> thr_bot
    seasons=["DJF","MAM","JJA","SON"],
    var = 'ENSO'
):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)

    for ax, season in zip(axes, seasons):
        enso = enso_seasonal[season]
        top_years = top_dict[season]
        bottom_years = bottom_dict[season]
        thr_top = thr_top_dict[season]
        thr_bot = thr_bot_dict[season]

        ax.plot(enso.time, enso, color="black", linewidth=1)

        # top 15%
        top_vals = enso.sel(time=top_years)
        ax.scatter(top_vals.time, top_vals, color="red", zorder=3)

        # bottom 15%
        bottom_vals = enso.sel(time=bottom_years)
        ax.scatter(bottom_vals.time, bottom_vals, color="blue", zorder=3)

        # thresholds
        ax.axhline(thr_top.item(), linestyle="--", color="red")
        ax.axhline(thr_bot.item(), linestyle="--", color="blue")
        ax.axhline(0, color="k", linewidth=0.8)

        ax.set_title(season, fontweight="bold")
        ax.set_xlabel("Year")

    axes[0].set_ylabel(f"{var} Index")
    plt.suptitle(f"{var} Top and Bottom 15%", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

def compute_and_plot_index_terciles_all_seasons(
    sst_seasonal_indices_LT,
    seasons=("DJF","MAM","JJA","SON"),
    pct=0.15,
    max_n=18,
    var = 'ENSO'
):
    """
    1) Groups ENSO by season
    2) Computes top/bottom tercile years per season
    3) Stores them in dicts
    4) Calls plot_enso_terciles_all_seasons
    """

    ENSO_SEASONAL = sst_seasonal_indices_LT.groupby("time.season")

    top_dict = {}
    bottom_dict = {}
    thr_top_dict = {}
    thr_bot_dict = {}

    for s in seasons:
        top_years, thr_top, bottom_years, thr_bot = get_tercile_years(
            ENSO_SEASONAL[s], pct=pct, max_n=max_n
        )
        top_dict[s] = top_years
        bottom_dict[s] = bottom_years
        thr_top_dict[s] = thr_top
        thr_bot_dict[s] = thr_bot

    plot_enso_terciles_all_seasons(
        ENSO_SEASONAL,
        top_dict, thr_top_dict,
        bottom_dict, thr_bot_dict,
        seasons=list(seasons), var = var,
    )
    
def get_tercile_years(enso, pct=0.15, max_n=18):
    n = enso.time.size
    k = min(int(np.ceil(n * pct)), max_n)

    sorted_desc = enso.sortby(enso, ascending=False)
    sorted_asc  = enso.sortby(enso, ascending=True)

    top_vals = sorted_desc.isel(time=slice(0, k))
    thr_top = top_vals.min()
    top_years = enso.time.where(enso >= thr_top, drop=True)

    bot_vals = sorted_asc.isel(time=slice(0, k))
    thr_bot = bot_vals.max()
    bot_years = enso.time.where(enso <= thr_bot, drop=True)

    return top_years, thr_top, bot_years, thr_bot

# def plot_enso_terciles_all_seasons(
#     enso_seasonal,   # e.g. sst_seasonal_indices_LT.ENSO.groupby('time.season')
#     top_dict,        # dict: season -> top_years
#     thr_top_dict,    # dict: season -> thr_top
#     bottom_dict,     # dict: season -> bottom_years
#     thr_bot_dict,    # dict: season -> thr_bot
#     seasons=["DJF","MAM","JJA","SON"],
#     var = 'ENSO'
# ):
#     fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)

#     for ax, season in zip(axes, seasons):
#         enso = enso_seasonal[season]
#         top_years = top_dict[season]
#         bottom_years = bottom_dict[season]
#         thr_top = thr_top_dict[season]
#         thr_bot = thr_bot_dict[season]

#         ax.plot(enso.time, enso, color="black", linewidth=1)

#         # top 15%
#         top_vals = enso.sel(time=top_years)
#         ax.scatter(top_vals.time, top_vals, color="red", zorder=3)

#         # bottom 15%
#         bottom_vals = enso.sel(time=bottom_years)
#         ax.scatter(bottom_vals.time, bottom_vals, color="blue", zorder=3)

#         # thresholds
#         ax.axhline(thr_top.item(), linestyle="--", color="red")
#         ax.axhline(thr_bot.item(), linestyle="--", color="blue")
#         ax.axhline(0, color="k", linewidth=0.8)

#         ax.set_title(season, fontweight="bold")
#         ax.set_xlabel("Year")

#     axes[0].set_ylabel(f"{var} Index")
#     plt.suptitle(f"{var} Top and Bottom 15%", fontsize=14, fontweight="bold")
#     plt.tight_layout()
#     plt.show()

# def plot_enso_terciles_all_seasons(
#     enso_seasonal,
#     top_dict,
#     thr_top_dict,
#     bottom_dict,
#     thr_bot_dict,
#     seasons=("DJF", "MAM", "JJA", "SON"),
#     var="ENSO",
# ):
#     """
#     Plot ENSO index time series with top/bottom 15% years highlighted.

#     Parameters
#     ----------
#     enso_seasonal : DatasetGroupBy
#         Seasonal ENSO index grouped by season, e.g.
#         ``sst_seasonal_indices_LT.ENSO.groupby('time.season')``.
#     top_dict : dict[str, array-like]
#         Season -> years in the top 15th percentile.
#     thr_top_dict : dict[str, scalar]
#         Season -> upper threshold value.
#     bottom_dict : dict[str, array-like]
#         Season -> years in the bottom 15th percentile.
#     thr_bot_dict : dict[str, scalar]
#         Season -> lower threshold value.
#     seasons : sequence of str, optional
#         Seasons to plot (default: DJF, MAM, JJA, SON).
#     var : str, optional
#         Variable label used in axis/title text (default: ``"ENSO"``).
#     """
#     COLORS = {"top": "#d62728", "bottom": "#1f77b4"}  # red / blue

#     fig, axes = plt.subplots(
#         1, len(seasons),
#         figsize=(4.5 * len(seasons), 4),
#         sharey=True,
#     )
#     fig.suptitle(
#         f"{var}",
#         fontsize=13,
#         fontweight="bold",
#         y=1.01,
#     )

#     for ax, season in zip(axes, seasons):
#         enso = enso_seasonal[season]
#         top_years = top_dict[season]
#         bottom_years = bottom_dict[season]
#         thr_top = float(thr_top_dict[season])
#         thr_bot = float(thr_bot_dict[season])

#         # --- time series ---
#         ax.plot(enso.time, enso, color="0.25", linewidth=0.9, zorder=1)
#         ax.axhline(0, color="0.6", linewidth=0.6, zorder=0)

#         # --- threshold bands ---
#         ax.axhline(thr_top, linestyle="--", linewidth=0.9,
#                    color=COLORS["top"], alpha=0.7)
#         ax.axhline(thr_bot, linestyle="--", linewidth=0.9,
#                    color=COLORS["bottom"], alpha=0.7)

#         # --- highlighted years ---
#         top_vals = enso.sel(time=top_years)
#         ax.scatter(top_vals.time, top_vals,
#                    color=COLORS["top"], s=20, zorder=3, label="Top 15%")

#         bot_vals = enso.sel(time=bottom_years)
#         ax.scatter(bot_vals.time, bot_vals,
#                    color=COLORS["bottom"], s=20, zorder=3, label="Bottom 15%")

#         # --- cosmetics ---
#         ax.set_title(season, fontweight="bold", pad=6)
#         ax.set_xlabel("Year", fontsize=9)
#         ax.tick_params(labelsize=8)
#         ax.spines[["top", "right"]].set_visible(False)

#     axes[0].set_ylabel(f"{var} index", fontsize=9)

#     # shared legend on the first panel only
#     handles, labels = axes[0].get_legend_handles_labels()
#     axes[0].legend(handles, labels, fontsize=7, frameon=False,
#                    loc="upper left", markerscale=1.2)

#     fig.tight_layout()
#     plt.show()

def plot_enso_terciles_all_seasons(
    enso_seasonal,
    top_dict,
    thr_top_dict,
    bottom_dict,
    thr_bot_dict,
    seasons=("DJF", "MAM", "JJA", "SON"),
    var="ENSO",
):
    """
    Plot ENSO index as bar charts with top/bottom 15% years highlighted.

    Parameters
    ----------
    enso_seasonal : DatasetGroupBy
        Seasonal ENSO index grouped by season.
    top_dict : dict[str, array-like]
        Season -> years in the top 15th percentile.
    thr_top_dict : dict[str, scalar]
        Season -> upper threshold value.
    bottom_dict : dict[str, array-like]
        Season -> years in the bottom 15th percentile.
    thr_bot_dict : dict[str, scalar]
        Season -> lower threshold value.
    seasons : sequence of str, optional
        Seasons to plot (default: DJF, MAM, JJA, SON).
    var : str, optional
        Variable label used in axis/title text (default: ``"ENSO"``).
    """
    COLORS = {
        "top":        "#b71c1c",  # dark red   — top 15%
        "bottom":     "#0d47a1",  # dark blue  — bottom 15%
        "pos":        "#ef9a9a",  # light red  — other positive years
        "neg":        "#90caf9",  # light blue — other negative years
        "threshold":  "0.35",
    }

    fig, axes = plt.subplots(
        1, len(seasons),
        figsize=(4.5 * len(seasons), 4),
        sharey=True,
    )
    fig.suptitle(
        f"{var} — top & bottom 15%",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for ax, season in zip(axes, seasons):
        enso = enso_seasonal[season]
        top_years    = set(top_dict[season].values)
        bottom_years = set(bottom_dict[season].values)
        thr_top = float(thr_top_dict[season])
        thr_bot = float(thr_bot_dict[season])

        times  = enso.time.values
        values = enso.values

        # assign a color to every bar
        bar_colors = []
        for t, v in zip(times, values):
            if t in top_years:
                bar_colors.append(COLORS["top"])
            elif t in bottom_years:
                bar_colors.append(COLORS["bottom"])
            elif v >= 0:
                bar_colors.append(COLORS["pos"])
            else:
                bar_colors.append(COLORS["neg"])

        ax.bar(times, values, color=bar_colors, width=0.8, zorder=2)

        # threshold lines
        ax.axhline(thr_top, linestyle="--", linewidth=0.9,
                   color=COLORS["top"], alpha=0.8, zorder=3)
        ax.axhline(thr_bot, linestyle="--", linewidth=0.9,
                   color=COLORS["bottom"], alpha=0.8, zorder=3)
        ax.axhline(0, color="0.5", linewidth=0.6, zorder=1)

        ax.set_title(season, fontweight="bold", pad=6)
        ax.set_xlabel("Year", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(f"{var} index", fontsize=9)

    # legend on first panel
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["top"],    label="Top 15%"),
        Patch(facecolor=COLORS["bottom"], label="Bottom 15%"),
        Patch(facecolor=COLORS["pos"],    label="Positive"),
        Patch(facecolor=COLORS["neg"],    label="Negative"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=7,
                   frameon=False, loc="upper left")

    fig.tight_layout()
    plt.show()

def compute_and_plot_index_terciles_all_seasons(
    sst_seasonal_indices_LT,
    seasons=("DJF","MAM","JJA","SON"),
    pct=0.15,
    max_n=18,
    var = 'ENSO'
):
    """
    1) Groups ENSO by season
    2) Computes top/bottom tercile years per season
    3) Stores them in dicts
    4) Calls plot_enso_terciles_all_seasons
    """

    ENSO_SEASONAL = sst_seasonal_indices_LT.groupby("time.season")

    top_dict = {}
    bottom_dict = {}
    thr_top_dict = {}
    thr_bot_dict = {}

    for s in seasons:
        top_years, thr_top, bottom_years, thr_bot = get_tercile_years(
            ENSO_SEASONAL[s], pct=pct, max_n=max_n
        )
        top_dict[s] = top_years
        bottom_dict[s] = bottom_years
        thr_top_dict[s] = thr_top
        thr_bot_dict[s] = thr_bot

    plot_enso_terciles_all_seasons(
        ENSO_SEASONAL,
        top_dict, thr_top_dict,
        bottom_dict, thr_bot_dict,
        seasons=list(seasons), var = var,
    )

# ---------- 1) ENSO top/bottom years (your existing function; included for completeness) ----------
def get_tercile_years(enso, pct=0.15, max_n=18):
    n = enso.time.size
    k = min(int(np.ceil(n * pct)), max_n)

    sorted_desc = enso.sortby(enso, ascending=False)
    sorted_asc  = enso.sortby(enso, ascending=True)

    top_vals = sorted_desc.isel(time=slice(0, k))
    thr_top = top_vals.min()
    top_years = enso.time.where(enso >= thr_top, drop=True)

    bot_vals = sorted_asc.isel(time=slice(0, k))
    thr_bot = bot_vals.max()
    bot_years = enso.time.where(enso <= thr_bot, drop=True)

    return top_years, thr_top, bot_years, thr_bot


# ---------- 2) Per-grid top/bottom extremes for 3D fields (PDSI + SST) ----------
def compute_grid_extremes(da, pct=0.15, max_n=18):
    n = da.sizes["time"]
    k = min(int(np.ceil(n * pct)), max_n)

    def kth_largest_1d(a, k):
        a = np.where(np.isfinite(a), a, -np.inf)
        return np.partition(a, len(a) - k)[len(a) - k]

    def kth_smallest_1d(a, k):
        a = np.where(np.isfinite(a), a, np.inf)
        return np.partition(a, k - 1)[k - 1]

    thr_top = xr.apply_ufunc(
        kth_largest_1d, da,
        kwargs={"k": k},
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    )

    thr_bot = xr.apply_ufunc(
        kth_smallest_1d, da,
        kwargs={"k": k},
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    )

    mask_top = da >= thr_top
    mask_bottom = da <= thr_bot
    return mask_top, mask_bottom, thr_top, thr_bot


def extract_for_enso_years(mask_top, mask_bottom, years):
    return mask_top.sel(time=years), mask_bottom.sel(time=years)


# ---------- 3) Agreement + overlap removal (shared for PDSI and SST) ----------
def compute_agreement_threshold_overlap_masks(top_sel, bottom_sel, threshold=3):
    agree_top = top_sel.sum(dim="time")
    agree_bottom = bottom_sel.sum(dim="time")

    top_gtN = agree_top.where(agree_top > threshold)
    bottom_gtN = agree_bottom.where(agree_bottom > threshold)

    overlap = (~top_gtN.isnull()) & (~bottom_gtN.isnull())

    top_final = top_gtN.where(~overlap)
    bottom_final = bottom_gtN.where(~overlap) * -1

    combined = xr.full_like(top_final, np.nan)
    combined = xr.where(top_final.notnull(), top_final, combined)
    combined = xr.where(bottom_final.notnull() & combined.isnull(), bottom_final, combined)

    return top_final, bottom_final, overlap, agree_top, agree_bottom, combined
