# Load Packages
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import xcdat as xc

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import zscore
from scipy.stats import t
from scipy.signal import butter, filtfilt, detrend, welch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from statsmodels.api import OLS, add_constant

def detrend_dim(da, dim):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=1)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def linear_reg_with_corr_p(pc_series, field, years):
    # Set matching time coordinates
    pc_series = pc_series.copy()
    field = field.copy()
    pc_series['time'] = years
    field['time'] = years

    # sample size
    n = field.sizes['time']
    dof = n - 2
    
    reg = xr.cov(field, pc_series, dim='time')

    # correlation map
    corr = xr.corr(field, pc_series, dim='time')

    # compute t-statistic
    tval = corr * np.sqrt(dof / (1 - corr**2))

    # compute two-tailed p-value
    p = xr.apply_ufunc(
        lambda x: 2 * (1 - t.cdf(np.abs(x), dof)),
        tval
    )

    return reg, corr, p

def four_reg_plot_with_sig(var, var_cmap, var_vval,
                           var_2, var_cmap_2, var_vval_2,
                           pvals1, pvals2,  # lists of p-value DataArrays
                           val_sel_titles, overall_title,
                           var_label1, var_label2, thresh = 0.05):
    robinson = ccrs.Robinson()

    # Use constrained layout to minimize whitespace automatically
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9),
        subplot_kw={'projection': robinson},
        constrained_layout=True
    )
    
        # fig, axes = plt.subplots(2, 2, figsize=(16, 8),
        #                      subplot_kw={'projection': robinson})
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    # Make colormaps with fixed bins once (avoids re-creating inside loop)
    cmap1 = plt.get_cmap(var_cmap, 10)
    cmap2 = plt.get_cmap(var_cmap_2, 10)

    for i, ax in enumerate(axes.flatten()):
        v1 = var[i]
        v2 = var_2[i]
        p1 = pvals1[i]
        p2 = pvals2[i]

        ax.coastlines(resolution='110m', linewidth=1.25)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle='--')

        # Gridlines + larger label size
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.025, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 16}   # bigger labels
        gl.ylabel_style = {'size': 16}

        # === Base data for var1
        lats = v1.lat
        lons = v1.lon
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        pcm1 = ax.pcolormesh(lon_grid, lat_grid, v1,
                             cmap=cmap1,
                             vmin=-var_vval, vmax=var_vval,
                             transform=ccrs.PlateCarree())

        # === Base data for var2
        lats2 = v2.lat
        lons2 = v2.lon
        lon_grid2, lat_grid2 = np.meshgrid(lons2, lats2)
        pcm2 = ax.pcolormesh(lon_grid2, lat_grid2, v2,
                             cmap=cmap2,
                             vmin=-var_vval_2, vmax=var_vval_2,
                             transform=ccrs.PlateCarree())

        # === Overlay significance mask for var1 (hatch non-sig)
        nonsig_mask1 = p1 > thresh
        ax.contourf(lon_grid, lat_grid, nonsig_mask1,
                    levels=[0.5, 1], colors='none', alpha = 0,
                    hatches=['..'], transform=ccrs.PlateCarree())

        # === Overlay significance mask for var2 (hatch non-sig)
        nonsig_mask2 = p2 > thresh
        ax.contourf(lon_grid2, lat_grid2, nonsig_mask2,
                    levels=[0.5, 1], colors='none', alpha = 0,
                    hatches=['..'], transform=ccrs.PlateCarree())

        ax.set_title(val_sel_titles[i], fontsize=22, y=1.0, fontweight="bold")

    # Shared colorbars inside the figure (no negative offsets), bigger ticks/labels
    # Left column (maps 0,2) uses var1 colorbar
    cb1 = fig.colorbar(
        pcm1, ax=axes[:, 0].ravel().tolist(),
        orientation='horizontal', fraction=0.06, pad=0.07, extend='both', shrink = 0.85
    )
    cb1.ax.tick_params(labelsize=14)        # bigger tick labels
    cb1.set_label(var_label1, fontsize=15)   # bigger cbar label

    # Right column (maps 1,3) uses var2 colorbar
    cb2 = fig.colorbar(
        pcm2, ax=axes[:, 1].ravel().tolist(),
        orientation='horizontal', fraction=0.06, pad=0.07, extend='both', shrink = 0.85
    )
    cb2.ax.tick_params(labelsize=14)
    cb2.set_label(var_label2, fontsize=15)

    # Compact, high-placed title
    fig.suptitle(overall_title, fontweight="bold", fontsize=20, y=1.05)
    plt.show()

def compute_and_plot_reg_with_corrs_p_MOV(idx_values,
                           pdsi_anoms_seasonal, 
                           sst_anoms_seasonal,
                           years, seasons, thresh,
                           cmap_pdsi='BrBG', vlim_pdsi=0.5,
                           cmap_sst='RdBu_r', vlim_sst=0.5,
                           overall_title = "Regession with Correlation P",
                           coeff_label1 = 'Regression Coefficient',
                            coeff_label2='Regression Coefficient'):
    pdsi_reg_dt = []
    sst_reg_dt = []
    pdsi_pvals_dt = []
    sst_pvals_dt = []
    dual_sel_titles = []
    thresh = thresh
    for season in seasons:
        reg_pdsi, corr_pdsi, p_pdsi = linear_reg_with_corr_p(
            idx_values[season],
            pdsi_anoms_seasonal[season],
            years
        )
        reg_sst, corr_sst, p_sst = linear_reg_with_corr_p(
            idx_values[season],
            sst_anoms_seasonal[season],
            years
        )
        pdsi_reg_dt.append(reg_pdsi)
        sst_reg_dt.append(reg_sst)
        pdsi_pvals_dt.append(p_pdsi)
        sst_pvals_dt.append(p_sst)
        title = f"{season}"
        dual_sel_titles.append(title)
    four_reg_plot_with_sig(
        pdsi_reg_dt, cmap_pdsi, vlim_pdsi,
        sst_reg_dt, cmap_sst, vlim_sst,
        pdsi_pvals_dt, sst_pvals_dt,
        dual_sel_titles,
        overall_title,
        coeff_label1, coeff_label2, thresh
    )
    return {
        'var1_reg': pdsi_reg_dt,
        'sst_reg': sst_reg_dt,
        'var1_pvals': pdsi_pvals_dt,
        'sst_pvals': sst_pvals_dt
    }

def plot_three_row_four_season_regressions(
    all_results,
    var1_titles,
    var1_labels,
    var1_cmaps, var1_vlims,
    sst_cmap, sst_vlim,
    seasons=['DJF','MAM','JJA','SON'],
    overall_title="Lagged Regressions"
):
    proj = ccrs.Robinson()
    fig, axes = plt.subplots(
        3, 4, figsize=(24, 15),
        subplot_kw={'projection': proj},
        constrained_layout=False
    )
    
    # Reduced hspace/wspace for tighter grouping
    plt.subplots_adjust(hspace=0.025, wspace=0.01, left=0.08, right=0.85, top=0.90, bottom=0.08)

    for row in range(3):
        results = all_results[row]
        vlim = var1_vlims[row]
        
        boundaries = np.round(np.linspace(-vlim, vlim, 11), 2)
        cmap1 = plt.get_cmap(var1_cmaps[row], len(boundaries) - 1)
        norm1 = mcolors.BoundaryNorm(boundaries, ncolors=cmap1.N, clip=False)

        for col, season in enumerate(seasons):
            ax = axes[row, col]
            ax.set_title(season if row == 0 else "", fontsize=20, fontweight='bold')
            ax.coastlines(resolution='110m', linewidth=0.8)

            # --- Strict Gridline Logic ---
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.3, linestyle='--')

            # Explicitly disable labels for internal plots
            gl.left_labels = (col == 0)
            gl.bottom_labels = True #(row == 2)
            gl.top_labels = False
            gl.right_labels = False
            # Formatting to ensure no "stray" labels
            gl.xlabel_style = {"size": 15}
            gl.ylabel_style = {"size": 15}
            
            var1 = results['var1_reg'][col]
            sst  = results['sst_reg'][col]
            pval1 = results['var1_pvals'][col]
            pval_sst = results['sst_pvals'][col]

            lon1, lat1 = np.meshgrid(var1.lon, var1.lat)
            pcm1 = ax.pcolormesh(lon1, lat1, var1, cmap=cmap1, norm=norm1, transform=ccrs.PlateCarree())

            # SST Setup
            sst_boundaries = np.round(np.linspace(-sst_vlim, sst_vlim, 11), 2)
            sst_cmap_obj = plt.get_cmap(sst_cmap, len(sst_boundaries) - 1)
            sst_norm = mcolors.BoundaryNorm(sst_boundaries, ncolors=sst_cmap_obj.N)
            
            lon2, lat2 = np.meshgrid(sst.lon, sst.lat)
            ax.pcolormesh(lon2, lat2, sst, cmap=sst_cmap_obj, norm=sst_norm, transform=ccrs.PlateCarree())

            # Significance
            ax.contourf(lon1, lat1, pval1 > 0.05, levels=[0.5, 1], colors='none', hatches=['..'], transform=ccrs.PlateCarree())
            ax.contourf(lon2, lat2, pval_sst > 0.05, levels=[0.5, 1], colors='none', hatches=['..'], transform=ccrs.PlateCarree())

        # Row Label
        axes[row, 0].annotate(var1_titles[row], xy=(-0.15, 0.5), xycoords='axes fraction',
                              rotation=90, va='center', ha='center', fontsize=20, fontweight='bold')

        # --- Var1 Colorbar: Under each row ---
        row_bbox_start = axes[row, 1].get_position()
        row_bbox_end = axes[row, 2].get_position()

        
        cbar_ax = fig.add_axes([
            row_bbox_start.x0,
            row_bbox_start.y0 - 0.065,   # was -0.045 — more negative = lower
            row_bbox_end.x1 - row_bbox_start.x0,
            0.018
        ])

        cb = fig.colorbar(pcm1, cax=cbar_ax, orientation='horizontal', extend="both", ticks=boundaries)
        cb.set_label(var1_labels[row], fontsize=18, labelpad=1)
        cb.ax.tick_params(labelsize=15)

    # --- SST Colorbar: Right Side ---
    top_pos = axes[0, -1].get_position()
    bot_pos = axes[2, -1].get_position()
    
    # Centered vertically relative to the plot area
    sst_cb_ax = fig.add_axes([0.88, bot_pos.y0, 0.015, top_pos.y1 - bot_pos.y0])
    sm_sst = plt.cm.ScalarMappable(cmap=sst_cmap_obj, norm=sst_norm)
    cb_sst = fig.colorbar(sm_sst, cax=sst_cb_ax, orientation='vertical', extend="both", ticks=sst_boundaries)
    cb_sst.ax.tick_params(labelsize=16)
    cb_sst.set_label("SST Reg. Coeff. [°C]", fontsize=18)#, fontweight='bold')

    fig.suptitle(overall_title, fontsize=25, weight="bold", y=0.95)
    plt.show()

# def plot_five_row_four_season_regressions(
#     all_results,
#     var1_titles,
#     var1_labels,
#     var1_cmaps, var1_vlims,
#     sst_cmap, sst_vlim,
#     seasons=['DJF','MAM','JJA','SON'],
#     overall_title="Lagged Regressions"
# ):
#     proj = ccrs.Robinson()
#     fig, axes = plt.subplots(
#         5, 4, figsize=(22, 18),
#         subplot_kw={'projection': proj},
#         constrained_layout=False
#     )
    
#     plt.subplots_adjust(hspace=0.15, wspace=0.01, left=0.08, right=0.85, top=0.93, bottom=0.05)

#     for row in range(5):
#         results = all_results[row]
#         vlim = var1_vlims[row]
        
#         boundaries = np.round(np.linspace(-vlim, vlim, 11), 2)
#         cmap1 = plt.get_cmap(var1_cmaps[row], len(boundaries) - 1)
#         norm1 = mcolors.BoundaryNorm(boundaries, ncolors=cmap1.N, clip=False)

#         for col, season in enumerate(seasons):
#             ax = axes[row, col]
#             ax.set_title(season if row == 0 else "", fontsize=16, fontweight='bold')
#             ax.coastlines(resolution='110m', linewidth=0.8)

#             gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.3, linestyle='--')
#             gl.left_labels = (col == 0)
#             gl.bottom_labels = True
#             gl.top_labels = False
#             gl.right_labels = False
#             gl.xlabel_style = {"size": 12}
#             gl.ylabel_style = {"size": 12}
            
#             var1 = results['var1_reg'][col]
#             sst  = results['sst_reg'][col]
#             pval1 = results['var1_pvals'][col]
#             pval_sst = results['sst_pvals'][col]

#             lon1, lat1 = np.meshgrid(var1.lon, var1.lat)
#             pcm1 = ax.pcolormesh(lon1, lat1, var1, cmap=cmap1, norm=norm1, transform=ccrs.PlateCarree())

#             sst_boundaries = np.round(np.linspace(-sst_vlim, sst_vlim, 11), 2)
#             sst_cmap_obj = plt.get_cmap(sst_cmap, len(sst_boundaries) - 1)
#             sst_norm = mcolors.BoundaryNorm(sst_boundaries, ncolors=sst_cmap_obj.N)
            
#             lon2, lat2 = np.meshgrid(sst.lon, sst.lat)
#             ax.pcolormesh(lon2, lat2, sst, cmap=sst_cmap_obj, norm=sst_norm, transform=ccrs.PlateCarree())

#             ax.contourf(lon1, lat1, pval1 > 0.05, levels=[0.5, 1], colors='none', hatches=['..'], transform=ccrs.PlateCarree())
#             ax.contourf(lon2, lat2, pval_sst > 0.05, levels=[0.5, 1], colors='none', hatches=['..'], transform=ccrs.PlateCarree())

#         # Row Label
#         axes[row, 0].annotate(var1_titles[row], xy=(-0.15, 0.5), xycoords='axes fraction',
#                               rotation=90, va='center', ha='center', fontsize=20, fontweight='bold')

#         # Var1 Colorbar
#         row_bbox_start = axes[row, 1].get_position()
#         row_bbox_end = axes[row, 2].get_position()
        
#         cbar_ax = fig.add_axes([
#             row_bbox_start.x0, 
#             row_bbox_start.y0 - 0.035, 
#             row_bbox_end.x1 - row_bbox_start.x0, 
#             0.015
#         ])
        
#         cb = fig.colorbar(pcm1, cax=cbar_ax, orientation='horizontal', extend="both", ticks=boundaries)
#         cb.set_label(var1_labels[row], fontsize=13, labelpad=1)
#         cb.ax.tick_params(labelsize=12)

#     # SST Colorbar
#     top_pos = axes[0, -1].get_position()
#     bot_pos = axes[4, -1].get_position()
    
#     sst_cb_ax = fig.add_axes([0.88, bot_pos.y0, 0.015, top_pos.y1 - bot_pos.y0])
#     sm_sst = plt.cm.ScalarMappable(cmap=sst_cmap_obj, norm=sst_norm)
#     cb_sst = fig.colorbar(sm_sst, cax=sst_cb_ax, orientation='vertical', extend="both", ticks=sst_boundaries)
#     cb_sst.ax.tick_params(labelsize=12)
#     cb_sst.set_label("SST Regression Coefficient [°C]", fontsize=13)

#     fig.suptitle(overall_title, fontsize=22, weight="bold", y=0.97)
#     plt.show()
    
#     ############ CCORRELLATTIOONN ########

def plot_five_row_four_season_regressions(
    all_results,
    var1_titles,
    var1_labels,
    var1_cmaps, var1_vlims,
    sst_cmap, sst_vlim,
    row0_sst_cmap=None, row0_sst_vlim=None,
    row0_var1_cmap=None, row0_var1_vlim=None,
    seasons=['DJF','MAM','JJA','SON'],
    overall_title="Lagged Regressions"
):
    # Fall back to shared defaults if row0 overrides not provided
    if row0_sst_cmap is None:  row0_sst_cmap  = sst_cmap
    if row0_sst_vlim is None:  row0_sst_vlim  = sst_vlim
    if row0_var1_cmap is None: row0_var1_cmap = var1_cmaps[0]
    if row0_var1_vlim is None: row0_var1_vlim = var1_vlims[0]

    proj = ccrs.Robinson()
    fig, axes = plt.subplots(
        5, 4, figsize=(25, 22),
        subplot_kw={'projection': proj},
        constrained_layout=False
    )

    # ── Manual layout in figure-fraction coordinates ──────────────────────
    left      = 0.08
    right     = 0.85
    top       = 0.93
    bottom    = 0.09
    wspace    = 0.01
    hspace    = 0.01
    extra_gap = 0.05

    n_rows, n_cols = 5, 4
    total_w = right - left
    ax_w    = (total_w - (n_cols - 1) * wspace) / n_cols

    total_h   = top - bottom
    gap_total = (n_rows - 1) * hspace + extra_gap
    ax_h      = (total_h - gap_total) / n_rows
    row0_h    = ax_h * 0.8   # row 0 is 80% the height of other rows — adjust to taste

    # Build y0 for each row from the bottom up.
    # The height of the row *below* r determines the step size.
    row_y0 = {}
    row_y0[4] = bottom
    for r in range(3, -1, -1):
        below_h = row0_h if (r + 1) == 0 else ax_h
        gap     = hspace + (extra_gap if r == 0 else 0)
        row_y0[r] = row_y0[r + 1] + below_h + gap

    # Apply positions — row 0 gets row0_h, all others get ax_h
    for r in range(n_rows):
        h = row0_h if r == 0 else ax_h
        for c in range(n_cols):
            x0 = left + c * (ax_w + wspace)
            axes[r, c].set_position([x0, row_y0[r], ax_w, h])
    # ──────────────────────────────────────────────────────────────────────

    for row in range(5):
        results = all_results[row]

        if row == 0:
            vlim  = row0_var1_vlim
            cmap1 = plt.get_cmap(row0_var1_cmap, 10)
        else:
            vlim  = var1_vlims[row]
            cmap1 = plt.get_cmap(var1_cmaps[row], 10)
        boundaries = np.round(np.linspace(-vlim, vlim, 11), 2)
        norm1 = mcolors.BoundaryNorm(boundaries, ncolors=cmap1.N, clip=False)

        if row == 0:
            row_sst_vlim = row0_sst_vlim
            row_sst_cmap = row0_sst_cmap
        else:
            row_sst_vlim = sst_vlim
            row_sst_cmap = sst_cmap
        sst_boundaries = np.round(np.linspace(-row_sst_vlim, row_sst_vlim, 11), 2)
        sst_cmap_obj   = plt.get_cmap(row_sst_cmap, len(sst_boundaries) - 1)
        sst_norm       = mcolors.BoundaryNorm(sst_boundaries, ncolors=sst_cmap_obj.N)

        for col, season in enumerate(seasons):
            ax = axes[row, col]
            ax.set_title(season if row == 0 else "", fontsize=16, fontweight='bold')
            ax.coastlines(resolution='110m', linewidth=1.25)
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.3, linestyle='--')
            gl.left_labels   = (col == 0)
            gl.bottom_labels = True
            gl.top_labels    = False
            gl.right_labels  = False
            gl.xlabel_style  = {"size": 12}
            gl.ylabel_style  = {"size": 12}

            var1     = results['var1_reg'][col]
            sst      = results['sst_reg'][col]
            pval1    = results['var1_pvals'][col]
            pval_sst = results['sst_pvals'][col]

            lon1, lat1 = np.meshgrid(var1.lon, var1.lat)
            pcm1 = ax.pcolormesh(lon1, lat1, var1, cmap=cmap1, norm=norm1, transform=ccrs.PlateCarree())

            lon2, lat2 = np.meshgrid(sst.lon, sst.lat)
            ax.pcolormesh(lon2, lat2, sst, cmap=sst_cmap_obj, norm=sst_norm, transform=ccrs.PlateCarree())

            ax.contourf(lon1, lat1, pval1    > 0.05, levels=[0.5, 1], colors='none', hatches=['...'], transform=ccrs.PlateCarree())
            ax.contourf(lon2, lat2, pval_sst > 0.05, levels=[0.5, 1], colors='none', hatches=['...'], transform=ccrs.PlateCarree())

        axes[row, 0].annotate(var1_titles[row], xy=(-0.15, 0.65), xycoords='axes fraction',
                              rotation=90, va='center', ha='center', fontsize=25, fontweight='bold')

        row_bbox_start = axes[row, 1].get_position()
        row_bbox_end   = axes[row, 2].get_position()
        total_width    = row_bbox_end.x1 - row_bbox_start.x0
        cbar_y         = row_bbox_start.y0 - 0.035

        if row == 0:
            gap       = 0.01
            bar_width = (total_width - gap) / 2

            cbar_ax = fig.add_axes([row_bbox_start.x0, cbar_y, bar_width, 0.015])
            cb = fig.colorbar(pcm1, cax=cbar_ax, orientation='horizontal', extend="both", ticks=boundaries)
            cb.set_label(var1_labels[row], fontsize=13, labelpad=1)
            cb.ax.tick_params(labelsize=12)

            sst_cbar_ax = fig.add_axes([row_bbox_start.x0 + bar_width + gap, cbar_y, bar_width, 0.015])
            sm_sst0 = plt.cm.ScalarMappable(cmap=sst_cmap_obj, norm=sst_norm)
            cb_sst0 = fig.colorbar(sm_sst0, cax=sst_cbar_ax, orientation='horizontal',
                                   extend="both", ticks=sst_boundaries)
            cb_sst0.set_label("Ocean Reg. Coeff [%]", fontsize=13, labelpad=1)
            cb_sst0.ax.tick_params(labelsize=12)

            row1_top         = axes[1, 0].get_position().y1
            row0_cbar_bottom = cbar_y
            sep_y = row1_top - 0.0017

            line = plt.Line2D(
                [0.06, 0.86], [sep_y, sep_y],
                transform=fig.transFigure,
                color='black', linewidth=1.5, linestyle='-',
                alpha=0.45, clip_on=False
            )
            fig.add_artist(line)

        else:
            cbar_ax = fig.add_axes([row_bbox_start.x0, cbar_y, total_width, 0.015])
            cb = fig.colorbar(pcm1, cax=cbar_ax, orientation='horizontal', extend="both", ticks=boundaries)
            cb.set_label(var1_labels[row], fontsize=13, labelpad=1)
            cb.ax.tick_params(labelsize=12)

    top_pos = axes[1, -1].get_position()
    bot_pos = axes[4, -1].get_position()

    shared_sst_boundaries = np.round(np.linspace(-sst_vlim, sst_vlim, 11), 2)
    shared_sst_cmap_obj   = plt.get_cmap(sst_cmap, len(shared_sst_boundaries) - 1)
    shared_sst_norm       = mcolors.BoundaryNorm(shared_sst_boundaries, ncolors=shared_sst_cmap_obj.N)

    sst_cb_ax = fig.add_axes([0.88, bot_pos.y0, 0.015, top_pos.y1 - bot_pos.y0])
    sm_sst    = plt.cm.ScalarMappable(cmap=shared_sst_cmap_obj, norm=shared_sst_norm)
    cb_sst    = fig.colorbar(sm_sst, cax=sst_cb_ax, orientation='vertical',
                             extend="both", ticks=shared_sst_boundaries)
    cb_sst.ax.tick_params(labelsize=12)
    cb_sst.set_label("SST Regression Coefficient [°C]", fontsize=13)

    fig.suptitle(overall_title, fontsize=22, weight="bold", y=0.95)
    plt.show()
    
    
def four_corr_plot_with_sig(var_corr, var_cmap, var_vval,
                            var2_corr, var_cmap_2, var_vval_2,
                            pvals1, pvals2,  # lists of p-value DataArrays
                            val_sel_titles, overall_title,
                            var_label='Correlation', thresh=0.05):
    """
    Plot 4 panels:
      - left column: var_corr (e.g., PDSI correlations)
      - right column: var2_corr (e.g., SST correlations)
    with hatching over NON-significant areas (p > thresh).
    """
    robinson = ccrs.Robinson()

    fig, axes = plt.subplots(
        2, 2, figsize=(15, 7.5),
        subplot_kw={'projection': robinson},
        constrained_layout=True
    )
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    # Fixed-bin colormaps
    cmap1 = plt.get_cmap(var_cmap, 10)
    cmap2 = plt.get_cmap(var_cmap_2, 10)

    for i, ax in enumerate(axes.flatten()):
        v1 = var_corr[i]
        v2 = var2_corr[i]
        p1 = pvals1[i]
        p2 = pvals2[i]

        ax.coastlines(resolution='110m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle='--')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.025, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}

        # === Base data for var1 (e.g., PDSI correlations)
        lats = v1.lat
        lons = v1.lon
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        pcm1 = ax.pcolormesh(lon_grid, lat_grid, v1,
                             cmap=cmap1,
                             vmin=-var_vval, vmax=var_vval,
                             transform=ccrs.PlateCarree())

        # === Base data for var2 (e.g., SST correlations)
        lats2 = v2.lat
        lons2 = v2.lon
        lon_grid2, lat_grid2 = np.meshgrid(lons2, lats2)
        pcm2 = ax.pcolormesh(lon_grid2, lat_grid2, v2,
                             cmap=cmap2,
                             vmin=-var_vval_2, vmax=var_vval_2,
                             transform=ccrs.PlateCarree())

        # === Overlay significance mask for var1 (hatch non-sig)
        nonsig_mask1 = p1 > thresh
        ax.contourf(lon_grid, lat_grid, nonsig_mask1,
                    levels=[0.5, 1], colors='none', alpha=0,
                    hatches=['..'], transform=ccrs.PlateCarree())

        # === Overlay significance mask for var2 (hatch non-sig)
        nonsig_mask2 = p2 > thresh
        ax.contourf(lon_grid2, lat_grid2, nonsig_mask2,
                    levels=[0.5, 1], colors='none', alpha=0,
                    hatches=['..'], transform=ccrs.PlateCarree())

        ax.set_title(val_sel_titles[i], fontsize=16, y=1.0, fontweight="bold")

    # Shared colorbars: left column for var1, right for var2
    cb1 = fig.colorbar(
        pcm1, ax=axes[:, 0].ravel().tolist(),
        orientation='horizontal', fraction=0.06, pad=0.07,
        extend='both', shrink=0.85
    )
    cb1.ax.tick_params(labelsize=14)
    cb1.set_label(var_label, fontsize=15)

    cb2 = fig.colorbar(
        pcm2, ax=axes[:, 1].ravel().tolist(),
        orientation='horizontal', fraction=0.06, pad=0.07,
        extend='both', shrink=0.85
    )
    cb2.ax.tick_params(labelsize=14)
    cb2.set_label(var_label, fontsize=15)

    fig.suptitle(overall_title, fontweight="bold", fontsize=18, y=1.05)
    plt.show()

def compute_and_plot_corr_with_p_MOV(idx_values,
                                     pdsi_anoms_seasonal,
                                     sst_anoms_seasonal,
                                     years, seasons, thresh,
                                     cmap_pdsi='BrBG', vlim_pdsi=0.6,
                                     cmap_sst='RdBu_r', vlim_sst=0.6,
                                     overall_title="Correlation with P-values",
                                     coeff_label='Correlation'):
    """
    For each season:
      - detrend index and fields
      - compute correlation and p-values with PDSI and SST
      - plot 2x2 panel of correlations + hatching where p > thresh
    """

    pdsi_corr_dt = []
    sst_corr_dt = []
    pdsi_pvals_dt = []
    sst_pvals_dt = []
    dual_sel_titles = []

    for season in seasons:
        # Correlation with PDSI
        reg_pdsi, corr_pdsi, p_pdsi = linear_reg_with_corr_p(
            detrend_dim(idx_values[season], 'time'),
            detrend_dim(pdsi_anoms_seasonal[season], 'time'),
            years
        )

        # Correlation with SST
        reg_sst, corr_sst, p_sst = linear_reg_with_corr_p(
            detrend_dim(idx_values[season], 'time'),
            detrend_dim(sst_anoms_seasonal[season], 'time'),
            years
        )

        # Store CORRELATIONS (not regressions)
        pdsi_corr_dt.append(corr_pdsi)
        sst_corr_dt.append(corr_sst)
        pdsi_pvals_dt.append(p_pdsi)
        sst_pvals_dt.append(p_sst)

        dual_sel_titles.append(f"{season}")

    # Plot correlations instead of regressions
    four_corr_plot_with_sig(
        pdsi_corr_dt, cmap_pdsi, vlim_pdsi,
        sst_corr_dt, cmap_sst, vlim_sst,
        pdsi_pvals_dt, sst_pvals_dt,
        dual_sel_titles,
        overall_title,
        coeff_label, thresh
    )

    return {
        'pdsi_corr': pdsi_corr_dt,
        'sst_corr': sst_corr_dt,
        'pdsi_pvals': pdsi_pvals_dt,
        'sst_pvals': sst_pvals_dt
    }


#### LONG VERSION ########

def four_reg_plot_with_sig_long(var, var_cmap, var_vval,
                           var_2, var_cmap_2, var_vval_2,
                           pvals1, pvals2,  # lists of p-value DataArrays
                           val_sel_titles, overall_title,
                           var_label1, var_label2, thresh = 0.05):
    robinson = ccrs.Robinson()

    # Use constrained layout to minimize whitespace automatically
    fig, axes = plt.subplots(
        1, 4, figsize=(22, 5.5),
        subplot_kw={'projection': robinson},
        constrained_layout=True
    )

    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    # Make colormaps with fixed bins once (avoids re-creating inside loop)
    cmap1 = plt.get_cmap(var_cmap, 10)
    cmap2 = plt.get_cmap(var_cmap_2, 10)

    for i, ax in enumerate(axes.flatten()):
        v1 = var[i]
        v2 = var_2[i]
        p1 = pvals1[i]
        p2 = pvals2[i]

        ax.coastlines(resolution='110m', linewidth=1.25)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle='--')

        # Gridlines + larger label size
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.025, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 16}   # bigger labels
        gl.ylabel_style = {'size': 16}

        # === Base data for var1
        lats = v1.lat
        lons = v1.lon
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        pcm1 = ax.pcolormesh(lon_grid, lat_grid, v1,
                             cmap=cmap1,
                             vmin=-var_vval, vmax=var_vval,
                             transform=ccrs.PlateCarree())

        # === Base data for var2
        lats2 = v2.lat
        lons2 = v2.lon
        lon_grid2, lat_grid2 = np.meshgrid(lons2, lats2)
        pcm2 = ax.pcolormesh(lon_grid2, lat_grid2, v2,
                             cmap=cmap2,
                             vmin=-var_vval_2, vmax=var_vval_2,
                             transform=ccrs.PlateCarree())

        # === Overlay significance mask for var1 (hatch non-sig)
        nonsig_mask1 = p1 > thresh
        ax.contourf(lon_grid, lat_grid, nonsig_mask1,
                    levels=[0.5, 1], colors='none', alpha = 0,
                    hatches=['..'], transform=ccrs.PlateCarree())

        # === Overlay significance mask for var2 (hatch non-sig)
        nonsig_mask2 = p2 > thresh
        ax.contourf(lon_grid2, lat_grid2, nonsig_mask2,
                    levels=[0.5, 1], colors='none', alpha = 0,
                    hatches=['..'], transform=ccrs.PlateCarree())

        ax.set_title(val_sel_titles[i], fontsize=22, y=1.0, fontweight="bold")

#     # Two shared horizontal colorbars below the single row, one per variable
#     cb1 = fig.colorbar(
#         pcm1, ax=axes.ravel().tolist(),
#         orientation='horizontal', fraction=0.05, pad=0.1, extend='both', shrink=0.4,
#         location='bottom', anchor=(0.0, 1.0)
#     )
#     cb1.ax.tick_params(labelsize=14)
#     cb1.set_label(var_label1, fontsize=15)

#     cb2 = fig.colorbar(
#         pcm2, ax=axes.ravel().tolist(),
#         orientation='horizontal', fraction=0.05, pad=0.1, extend='both', shrink=0.4,
#         location='bottom', anchor=(1.0, 1.0)
#     )
#     cb2.ax.tick_params(labelsize=14)
#     cb2.set_label(var_label2, fontsize=15)

    cb1 = fig.colorbar(
        pcm1, ax=axes[0:2].tolist(),
        orientation='horizontal', fraction=0.05, pad=0.1, extend='both', shrink=0.95,
        location='bottom'
    )
    cb1.ax.tick_params(labelsize=14)
    cb1.set_label(var_label1, fontsize=15)

    cb2 = fig.colorbar(
        pcm2, ax=axes[2:4].tolist(),
        orientation='horizontal', fraction=0.05, pad=0.1, extend='both', shrink=0.95,
        location='bottom'
    )
    cb2.ax.tick_params(labelsize=14)
    cb2.set_label(var_label2, fontsize=15)

    fig.suptitle(overall_title, fontweight="bold", fontsize=20, y=0.98)
    plt.show()
    
def compute_and_plot_reg_with_corrs_p_MOV_long(idx_values,
                           pdsi_anoms_seasonal, 
                           sst_anoms_seasonal,
                           years, seasons, thresh,
                           cmap_pdsi='BrBG', vlim_pdsi=0.5,
                           cmap_sst='RdBu_r', vlim_sst=0.5,
                           overall_title = "Regession with Correlation P",
                           coeff_label1 = 'Regression Coefficient',
                            coeff_label2='Regression Coefficient'):
    pdsi_reg_dt = []
    sst_reg_dt = []
    pdsi_pvals_dt = []
    sst_pvals_dt = []
    dual_sel_titles = []
    thresh = thresh
    for season in seasons:
        reg_pdsi, corr_pdsi, p_pdsi = linear_reg_with_corr_p(
            idx_values[season],
            pdsi_anoms_seasonal[season],
            years
        )
        reg_sst, corr_sst, p_sst = linear_reg_with_corr_p(
            idx_values[season],
            sst_anoms_seasonal[season],
            years
        )
        pdsi_reg_dt.append(reg_pdsi)
        sst_reg_dt.append(reg_sst)
        pdsi_pvals_dt.append(p_pdsi)
        sst_pvals_dt.append(p_sst)
        title = f"{season}"
        dual_sel_titles.append(title)
    four_reg_plot_with_sig_long(
        pdsi_reg_dt, cmap_pdsi, vlim_pdsi,
        sst_reg_dt, cmap_sst, vlim_sst,
        pdsi_pvals_dt, sst_pvals_dt,
        dual_sel_titles,
        overall_title,
        coeff_label1, coeff_label2, thresh
    )
    return {
        'var1_reg': pdsi_reg_dt,
        'sst_reg': sst_reg_dt,
        'var1_pvals': pdsi_pvals_dt,
        'sst_pvals': sst_pvals_dt
    }

### NORTH AMERICAN ONLY ####

def four_reg_plot_with_sig_long_NA(var, var_cmap, var_vval,
                           var_2, var_cmap_2, var_vval_2,
                           pvals1, pvals2,  # lists of p-value DataArrays
                           val_sel_titles, overall_title,
                           var_label1, var_label2, thresh = 0.05):
    proj = ccrs.PlateCarree()
    na_extent = [-170, -50, 5, 75]  # lon_min, lon_max, lat_min, lat_max

    fig, axes = plt.subplots(
        1, 4, figsize=(20, 5.0),
        subplot_kw={'projection': proj},
        constrained_layout=True
    )
    cmap1 = plt.get_cmap(var_cmap, 10)
    cmap2 = plt.get_cmap(var_cmap_2, 10)

    for i, ax in enumerate(axes.flatten()):
        v1 = var[i]
        v2 = var_2[i]
        p1 = pvals1[i]
        p2 = pvals2[i]

        ax.set_extent(na_extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='110m', linewidth=1.25)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle='--')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.4, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}

        lats = v1.lat
        lons = v1.lon
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        pcm1 = ax.pcolormesh(lon_grid, lat_grid, v1,
                             cmap=cmap1,
                             vmin=-var_vval, vmax=var_vval,
                             transform=ccrs.PlateCarree())

        lats2 = v2.lat
        lons2 = v2.lon
        lon_grid2, lat_grid2 = np.meshgrid(lons2, lats2)
        pcm2 = ax.pcolormesh(lon_grid2, lat_grid2, v2,
                             cmap=cmap2,
                             vmin=-var_vval_2, vmax=var_vval_2,
                             transform=ccrs.PlateCarree())

        nonsig_mask1 = p1 > thresh
        ax.contourf(lon_grid, lat_grid, nonsig_mask1,
                    levels=[0.5, 1], colors='none', alpha = 0,
                    hatches=['..'], transform=ccrs.PlateCarree())

        nonsig_mask2 = p2 > thresh
        ax.contourf(lon_grid2, lat_grid2, nonsig_mask2,
                    levels=[0.5, 1], colors='none', alpha = 0,
                    hatches=['..'], transform=ccrs.PlateCarree())

        ax.set_title(val_sel_titles[i], fontsize=22, y=1.0, fontweight="bold")

    cb1 = fig.colorbar(
        pcm1, ax=axes[0:2].tolist(),
        orientation='horizontal', fraction=0.05, pad=0.1, extend='both', shrink=0.95,
        location='bottom'
    )
    cb1.ax.tick_params(labelsize=14)
    cb1.set_label(var_label1, fontsize=15)

    cb2 = fig.colorbar(
        pcm2, ax=axes[2:4].tolist(),
        orientation='horizontal', fraction=0.05, pad=0.1, extend='both', shrink=0.95,
        location='bottom'
    )
    cb2.ax.tick_params(labelsize=14)
    cb2.set_label(var_label2, fontsize=15)

    fig.suptitle(overall_title, fontweight="bold", fontsize=20, y=1.05)
    plt.show()
    
    
def compute_and_plot_reg_with_corrs_p_MOV_NA(idx_values,
                           pdsi_anoms_seasonal, 
                           sst_anoms_seasonal,
                           years, seasons, thresh,
                           cmap_pdsi='BrBG', vlim_pdsi=0.5,
                           cmap_sst='RdBu_r', vlim_sst=0.5,
                           overall_title = "Regession with Correlation P",
                           coeff_label1 = 'Regression Coefficient',
                            coeff_label2='Regression Coefficient'):
    pdsi_reg_dt = []
    sst_reg_dt = []
    pdsi_pvals_dt = []
    sst_pvals_dt = []
    dual_sel_titles = []
    thresh = thresh
    for season in seasons:
        reg_pdsi, corr_pdsi, p_pdsi = linear_reg_with_corr_p(
            idx_values[season],
            pdsi_anoms_seasonal[season],
            years
        )
        reg_sst, corr_sst, p_sst = linear_reg_with_corr_p(
            idx_values[season],
            sst_anoms_seasonal[season],
            years
        )
        pdsi_reg_dt.append(reg_pdsi)
        sst_reg_dt.append(reg_sst)
        pdsi_pvals_dt.append(p_pdsi)
        sst_pvals_dt.append(p_sst)
        title = f"{season}"
        dual_sel_titles.append(title)
    
    four_reg_plot_with_sig_long_NA(
        pdsi_reg_dt, cmap_pdsi, vlim_pdsi,
        sst_reg_dt, cmap_sst, vlim_sst,
        pdsi_pvals_dt, sst_pvals_dt,
        dual_sel_titles,
        overall_title,
        coeff_label1, coeff_label2, thresh
    )
    return {
        'var1_reg': pdsi_reg_dt,
        'sst_reg': sst_reg_dt,
        'var1_pvals': pdsi_pvals_dt,
        'sst_pvals': sst_pvals_dt
    }

# def compute_and_plot_reg_with_corrs_p_MOV_wo_DT(idx_values,
#                            pdsi_anoms_seasonal, 
#                            sst_anoms_seasonal,
#                            years, seasons, thresh,
#                            cmap_pdsi='BrBG', vlim_pdsi=0.5,
#                            cmap_sst='RdBu_r', vlim_sst=0.5,
#                            overall_title = "Regession with Correlation P",
#                            coeff_label1 = 'Regression Coefficient',
#                             coeff_label2='Regression Coefficient'):
#     pdsi_reg_dt = []
#     sst_reg_dt = []
#     pdsi_pvals_dt = []
#     sst_pvals_dt = []
#     dual_sel_titles = []
#     thresh = thresh
#     for season in seasons:
#         reg_pdsi, corr_pdsi, p_pdsi = linear_reg_with_corr_p(
#             detrend_dim(idx_values[season], 'time'),
#             detrend_dim(pdsi_anoms_seasonal[season], 'time'),
#             years
#         )
#         reg_sst, corr_sst, p_sst = linear_reg_with_corr_p(
#             detrend_dim(idx_values[season], 'time'),
#             detrend_dim(sst_anoms_seasonal[season], 'time'),
#             years
#         )
#         pdsi_reg_dt.append(reg_pdsi)
#         sst_reg_dt.append(reg_sst)
#         pdsi_pvals_dt.append(p_pdsi)
#         sst_pvals_dt.append(p_sst)
#         title = f"{season}"
#         dual_sel_titles.append(title)
#     four_reg_plot_with_sig(
#         pdsi_reg_dt, cmap_pdsi, vlim_pdsi,
#         sst_reg_dt, cmap_sst, vlim_sst,
#         pdsi_pvals_dt, sst_pvals_dt,
#         dual_sel_titles,
#         overall_title,
#         coeff_label1, coeff_label2, thresh
#     )
#     return {
#         'var1_reg': pdsi_reg_dt,
#         'sst_reg': sst_reg_dt,
#         'var1_pvals': pdsi_pvals_dt,
#         'sst_pvals': sst_pvals_dt
#     }

