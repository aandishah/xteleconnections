

def process_geopotential_pressure_level(z, pressure_level= None, season= None, time_range=None, lat_range=None, lon_range=None):

    g = 9.81  # gravity (m/s^2)

    # Load and preprocess dataset
    #Z = xr.open_dataset(filepath).drop_vars('number', errors='ignore')
    Z = z
    Z = Z.rename({dim: new_name for dim, new_name in {'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'}.items() if dim in Z.dims})
    Z = Z.assign_coords(time=pd.to_datetime(Z["time"].values, format="%Y%m%d"))
    Z = Z.drop_vars('expver', errors='ignore')
    Z = Z.sel(time=slice(*time_range))

    if lat_range:
        Z = Z.sel(lat=slice(*lat_range))

    # # Select variable at given pressure level and convert to geopotential height (m)
    Z_level = Z.sel(pressure_level=pressure_level)['z']
    h = Z_level / g     # Convert to geopotential height
    h_var_name = f"h_{pressure_level}"

    ds_out = xr.Dataset({
        'z': Z_level,
        h_var_name: h
    })
    
#     h_500 = process_geopotential_pressure_level(
#     z,
#     pressure_level=500,
#     time_range = ('1950-01-01', '2023-12-31'),
#     lat_range=(90, -90),
#     )

#     h_500 = h_500.bounds.add_missing_bounds()

    return ds_out