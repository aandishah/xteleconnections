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
    

# -*- Plotting # -*-

def single_plot( var_select, overall_title,
               var_cmap, split, var_vval, var_vval_2, var_label):
    # Define the Robinson projection

    robinson = ccrs.Robinson()

    # Create 1x1 grid of subplots with the Robinson projection and adjust hspace
    fig, axes = plt.subplots(1, 1, figsize=(8, 10), subplot_kw={'projection': robinson}) 

    axes.coastlines(resolution='110m', linewidth=0.5)
    axes.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle='--')
    axes.gridlines()

            # Create a grid of lats and lons for pcolormesh

    lats = var_select.lat
    lons = var_select.lon
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    gl = axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1, color='gray', alpha=0.025, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

            # Create Discrete Color Map
            #cmap = var_cmap
    cmap = plt.get_cmap(var_cmap, split) 

            # !!!!!!!!!!!!!!!! Plot Data on Map !!!!!!!!!!!!!!!!
    pcm = axes.pcolormesh(lon_grid, lat_grid, var_select, 
                                cmap=cmap, vmin = (var_vval), vmax = var_vval_2, 
                                transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(
        pcm,
        ax=axes,
        orientation='horizontal',
        location='bottom', 
        shrink=0.7,
        pad=0.05,
        extend = "both"
        #ticks=ticks
    )
    
    cbar.set_label(var_label, fontsize=12)
    plt.title(overall_title, fontweight="bold", fontsize = 15) # , x=0.25, y=1.1)
    plt.show()
