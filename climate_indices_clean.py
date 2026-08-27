"""
climate_indices_clean.py
=========================

Style- and structure-unified rewrite of the climate-index functions from
index_prep.ipynb:

    compute_enso, compute_iod, compute_amo, compute_tpi,
    compute_pdo, calculate_pdo_ncl,
    compute_aao, compute_ao, compute_sam, compute_nao,
    compute_npi

See CHANGES.md (delivered alongside this file) for a full, bolded list of
what changed in each function and why, plus a few things that were found
but deliberately left alone because fixing them would change numeric
results and could not be verified without a code-execution environment.

Every function keeps its original scientific definition (region boxes,
formulas, EOF/SVD math). What changed is structure and style:
  - one shared set of helpers for the logic that was copy-pasted (with
    small inconsistent variations) across most of the original functions
  - consistent numpydoc-style docstrings
  - consistent parameter naming (clim_start/clim_end, lat_bounds/lon_bounds)
  - consistent index naming and metadata attrs via `_set_index_attrs`
"""

import numpy as np
import xarray as xr
from scipy.signal import detrend, firwin
from eofs.standard import Eof


# ============================================================================
# Shared helpers
# ============================================================================

def _lat_slice(da, lat_min, lat_max, lat_dim="lat"):
    """
    Build a `slice(...)` for [lat_min, lat_max] that respects whether
    `da`'s latitude coordinate is ascending or descending.

    Several of the original functions hard-coded one order or the other
    (e.g. always `slice(lat_max, lat_min)`), which silently returns an
    empty selection if the assumption is wrong. This is used everywhere
    a lat box is selected so the behavior is uniform and order-safe.
    """
    if da[lat_dim][0] > da[lat_dim][-1]:
        return slice(lat_max, lat_min)
    return slice(lat_min, lat_max)


def _region_slice(da, lat_bounds, lon_bounds, lat_dim="lat", lon_dim="lon"):
    """
    Select a lat/lon box from `da`, safe for either latitude order and
    for longitude boxes that cross the antimeridian (pass `lon_min >
    lon_max`, e.g. `(140, -145)`, to select 140E through 145W the short
    way around).
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    region = da.sel({lat_dim: _lat_slice(da, lat_min, lat_max, lat_dim)})

    if lon_min <= lon_max:
        return region.sel({lon_dim: slice(lon_min, lon_max)})

    east = region.sel({lon_dim: slice(lon_min, 180)})
    west = region.sel({lon_dim: slice(-180, lon_max)})
    return xr.concat([east, west], dim=lon_dim)


def _region_mean(da, lat_bounds, lon_bounds, lat_dim="lat", lon_dim="lon"):
    """Area-weighted (cos-lat) mean of `da` over a lat/lon box."""
    region = _region_slice(da, lat_bounds, lon_bounds, lat_dim, lon_dim)
    weights = np.cos(np.deg2rad(region[lat_dim]))
    weights.name = "weights"
    return region.weighted(weights).mean(dim=[lat_dim, lon_dim])


def _monthly_anomalies(da, clim_start=None, clim_end=None, dim="time"):
    """
    Remove the monthly climatology from `da`.

    The climatology is computed from the `[clim_start, clim_end]` window
    (either bound may be None to leave that side open; both None uses the
    full record) and subtracted from the *full* time series, so the
    returned anomalies always span all of `da`'s time axis. The leftover
    `month` coordinate that `groupby` adds is always dropped, which some
    of the original functions did inconsistently.
    """
    if clim_start is not None or clim_end is not None:
        base = da.sel({dim: slice(clim_start, clim_end)})
    else:
        base = da
    climatology = base.groupby(f"{dim}.month").mean(dim)
    anomalies = da.groupby(f"{dim}.month") - climatology
    if "month" in anomalies.coords:
        anomalies = anomalies.drop_vars("month")
    return anomalies


def _standardize(da, base_start=None, base_end=None, dim="time"):
    """
    Zero-mean, unit-variance standardization. If `base_start`/`base_end`
    are given, the mean and std are computed only over that window (e.g.
    a fixed reference period) and applied to the full series; otherwise
    the full series' own mean/std is used.
    """
    if base_start is not None or base_end is not None:
        base = da.sel({dim: slice(base_start, base_end)})
    else:
        base = da
    return (da - base.mean(dim)) / base.std(dim)


def _scale_by_std(da, base_start=None, base_end=None, dim="time"):
    """
    Divide by the standard deviation over a base period, with **no** mean
    subtraction. Used where a PC1 series is already ~zero-mean from the
    EOF decomposition itself and the original code only ever rescaled it
    (AAO, AO) rather than re-centering it — that's a different operation
    from `_standardize` above, kept separate rather than folded together
    so each function's original numeric behavior is preserved exactly.
    """
    base = da.sel({dim: slice(base_start, base_end)}) if (base_start or base_end) else da
    return da / base.std(dim)


def _standardize_by_month(da, clim_start, clim_end, dim="time"):
    """
    Standardize each calendar month separately: subtract that month's
    base-period mean and divide by that month's base-period std. Used by
    `compute_sam`, which standardizes month-by-month rather than removing
    a climatology and standardizing the anomaly series as a whole.
    """
    base = da.sel({dim: slice(str(clim_start), str(clim_end))})
    month_mean = base.groupby(f"{dim}.month").mean()
    month_std = base.groupby(f"{dim}.month").std()
    return xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        da.groupby(f"{dim}.month"), month_mean, month_std,
    )


def _flip_sign_at_point(eof1, pc1, lat_pt, lon_pt, lat_dim="lat", lon_dim="lon", positive=True):
    """
    Flip the sign of an (eof1, pc1) pair so that `eof1`'s value nearest
    (lat_pt, lon_pt) has the requested sign. Used for the EOF-based
    indices, each of which defines its sign convention at a specific
    point (e.g. PDO at 37N/160W).
    """
    val = float(eof1.sel({lat_dim: lat_pt, lon_dim: lon_pt}, method="nearest"))
    if (positive and val < 0) or (not positive and val > 0):
        return -eof1, -pc1
    return eof1, pc1


def _set_index_attrs(da, short_name, long_name, description, units="", **extra):
    """
    Apply a consistent name + attrs block to a finished index DataArray.
    Every index below ends up with the same attrs keys
    (long_name/description/units, plus whatever extra metadata is
    relevant), which the originals did inconsistently (some had no attrs
    at all, some had partial attrs, naming varied between "NAO_index",
    "SAM_Index", "ao", "AAO", etc.).
    """
    da.name = short_name
    da.attrs.update({"long_name": long_name, "description": description, "units": units, **extra})
    return da


# ============================================================================
# SST box / anomaly indices: ENSO, AMO, IOD, TPI
# ============================================================================

def compute_enso(ds, sst_var="sst", lat_bounds=(-5, 5), lon_bounds=(-170, -120),
                  clim_start="1981", clim_end="2010", smooth_window=None):
    """
    Compute the ENSO (Nino 3.4) index from SST data.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Sea surface temperature. If a Dataset, `sst_var` is extracted.
    sst_var : str, default "sst"
        Name of the SST variable when `ds` is a Dataset.
    lat_bounds, lon_bounds : tuple of float
        Box defining the Nino 3.4 region (default: 5S-5N, 170W-120W).
    clim_start, clim_end : str or None
        Climatology reference period. Both None uses the full record.
    smooth_window : int, optional
        Rolling-mean window (months) applied to the anomalies. Typically
        3 months for ENSO. None or 0 skips smoothing.

    Returns
    -------
    xr.DataArray
        The ENSO Nino 3.4 index time series.
    """
    sst = ds[sst_var] if isinstance(ds, xr.Dataset) else ds

    sst_ts = _region_mean(sst, lat_bounds, lon_bounds)
    anomalies = _monthly_anomalies(sst_ts, clim_start, clim_end)

    index = (anomalies.rolling(time=smooth_window, center=True).mean(min_periods=smooth_window)
             if smooth_window else anomalies)

    return _set_index_attrs(
        index, "ENSO", "ENSO Nino 3.4 Index",
        "Area-weighted SST anomalies in the Nino 3.4 region",
        units="degC",
        smoothing=f"{smooth_window}-month running mean" if smooth_window else "None",
    )


def compute_amo(ds, sst_var="sst", lat_bounds=(0, 60), lon_bounds=(-80, 0),
                 clim_start=None, clim_end=None, smooth_window=None):
    """
    Compute the Atlantic Multidecadal Oscillation (AMO) index from SST
    data. Directly translated from the NCL AMO implementation.

    Parameters
    ----------
    ds, sst_var, lat_bounds, lon_bounds : see `compute_enso`.
        Default region: North Atlantic, 0-60N, 80W-0.
    clim_start, clim_end : str or None
        Climatology reference period. Both None uses the full record.
    smooth_window : int, optional
        Rolling-mean window (months). Typically 121 (10 years) for AMO.

    Returns
    -------
    xr.DataArray
        The detrended, area-weighted AMO index time series.

    Notes
    -----
    Unlike `compute_enso`, AMO removes a linear trend (the global-warming
    signal) from the anomalies before optional smoothing. That is part of
    AMO's standard definition, not a style inconsistency with ENSO.
    """
    sst = ds[sst_var] if isinstance(ds, xr.Dataset) else ds

    na_sst_mean = _region_mean(sst, lat_bounds, lon_bounds)
    anomalies = _monthly_anomalies(na_sst_mean, clim_start, clim_end)

    valid = anomalies.dropna(dim="time")
    detrended_vals = detrend(valid.values)
    detrended = xr.DataArray(detrended_vals, coords={"time": valid.time}, dims=["time"])
    detrended = detrended.reindex(time=anomalies.time)

    index = detrended.rolling(time=smooth_window, center=True).mean() if smooth_window else detrended

    return _set_index_attrs(
        index, "AMO", "Atlantic Multidecadal Oscillation Index",
        "Area-weighted, detrended North Atlantic SST anomalies",
        units="degC",
        smoothing=f"{smooth_window}-month running mean" if smooth_window else "None",
    )


def compute_iod(sst, clim_start="1991", clim_end="2020", remove_global_mean=True):
    """
    Compute the Dipole Mode Index (DMI / IOD) from raw SST data.

    Parameters
    ----------
    sst : xr.DataArray
        Sea surface temperature, dims (time, lat, lon).
    clim_start, clim_end : str or None
        Climatology reference period. Both None uses the full record.
    remove_global_mean : bool, default True
        If True, subtract the area-weighted global-mean anomaly from
        every grid cell before computing the west/east boxes.

    Returns
    -------
    xr.DataArray
        IOD index: west box SST anomaly minus east box SST anomaly.
    """
    anomalies = _monthly_anomalies(sst, clim_start, clim_end)

    if remove_global_mean:
        global_mean = _region_mean(anomalies, (-90, 90), (-180, 180))
        anomalies = anomalies - global_mean

    # IOD boxes (Saji et al. 1999) — plain (unweighted) box means, matching
    # the original function; only the global-mean removal above is
    # area-weighted.
    west = _region_slice(anomalies, (-10, 10), (50, 70)).mean(dim=["lat", "lon"])   # 50-70E, 10S-10N
    east = _region_slice(anomalies, (-10, 0), (90, 110)).mean(dim=["lat", "lon"])   # 90-110E, 10S-0

    index = west - east

    return _set_index_attrs(
        index, "IOD", "Dipole Mode Index (West - East SST)",
        "Computed with monthly climatology removal and global-mean removal.",
        units="degC",
    )


def compute_tpi(sst, clim_start="1971-01", clim_end="2000-12", apply_filter=False):
    """
    Compute the Tripole Index (TPI) for the Interdecadal Pacific
    Oscillation, per Henley et al. (2015).

    Regions: T1 25-45N/140E-145W, T2 10S-10N/170E-90W, T3 50-15S/150E-160W.

    Parameters
    ----------
    sst : xr.DataArray
        SST, dims (time, lat, lon), lon in [-180, 180].
    clim_start, clim_end : str or None
        Climatology reference period. Both None uses the full record.
    apply_filter : bool, default False
        If True, apply a 13-year Lanczos low-pass filter to the raw TPI.

    Returns
    -------
    xr.DataArray
        TPI index time series (T2 minus the mean of T1 and T3).
    """
    anomalies = _monthly_anomalies(sst, clim_start, clim_end)

    T1 = _region_mean(anomalies, (25, 45), (140, -145))
    T2 = _region_mean(anomalies, (-10, 10), (170, -90))
    T3 = _region_mean(anomalies, (-50, -15), (150, -160))

    index = T2 - (T1 + T3) / 2.0
    index = _set_index_attrs(
        index, "TPI", "Tripole Index (IPO)",
        "T2 minus the mean of T1 and T3 area-weighted SST anomalies (Henley et al. 2015).",
        units="degC",
    )

    if apply_filter:
        n, fc = 157, 1.0 / 156.0
        half = (n - 1) // 2
        wf = firwin(n, fc, window="lanczos")
        v = index.values.astype(float)
        filtered = np.array([
            np.dot(wf, v[i - half: i + half + 1])
            if half <= i < len(v) - half and not np.any(np.isnan(v[i - half: i + half + 1]))
            else np.nan
            for i in range(len(v))
        ])
        index = index.copy(data=filtered)
        index.attrs["smoothing"] = "13-year Lanczos low-pass filter"
    else:
        index.attrs["smoothing"] = "None"

    return index


# ============================================================================
# EOF-based indices: PDO (two variants), AAO, AO, NAO
# ============================================================================

def compute_pdo(sst, target, lat_bounds=(20, 70), lon_bounds=(110, -100),
                 clim_start="1920-01-01", clim_end="2014-12-31"):
    """
    Compute the PDO index and spatial pattern from SST via EOF analysis.

    Steps: monthly climatology removal -> global-mean (60S-70N) removal
    -> leading EOF of North Pacific SST anomalies -> sign convention at
    37N/160W -> standardize by the climatology-period std.

    Parameters
    ----------
    sst : xr.DataArray
        SST, dims (time, lat, lon). `sst['time']` is overwritten with
        `target['time']` before processing (see Notes).
    target : xr.Dataset or xr.DataArray
        Source of the time coordinate `sst` is aligned to, and of the
        time axis for the returned index.
    lat_bounds, lon_bounds : tuple of float
        North Pacific EOF domain (default: 20-70N, 110E-100W).
    clim_start, clim_end : str or None
        Climatology and standardization reference period.

    Returns
    -------
    pdo_index : xr.DataArray
        Standardized PDO index time series.
    pdo_pattern : xr.DataArray
        Leading EOF spatial pattern (lat, lon), sign-corrected.

    Notes
    -----
    `sst['time']` is overwritten with `target['time']` because the two
    inputs are assumed to represent the same months on different time
    encodings; this preserves the original function's behavior.
    """
    sst = sst.copy()
    sst["time"] = target["time"]

    anomalies = _monthly_anomalies(sst, clim_start, clim_end)

    global_mean = _region_mean(anomalies, (-60, 70), (-180, 180))
    anomalies = anomalies - global_mean

    pac = _region_slice(anomalies, lat_bounds, lon_bounds)
    if np.all(np.isnan(pac.values)):
        raise ValueError("The subsetted North Pacific region contains only NaNs.")

    coslat = np.cos(np.deg2rad(pac.lat.values))
    wgts = np.sqrt(coslat)[..., np.newaxis]

    solver = Eof(pac.transpose("time", "lat", "lon").values, weights=wgts)
    eof1 = xr.DataArray(solver.eofs(neofs=1)[0], coords={"lat": pac.lat, "lon": pac.lon}, dims=["lat", "lon"])
    pc1 = solver.pcs(npcs=1, pcscaling=1)[:, 0]

    eof1, pc1 = _flip_sign_at_point(eof1, pc1, lat_pt=37, lon_pt=-160, positive=True)

    pc1_da = xr.DataArray(pc1, coords={"time": pac.time}, dims=["time"])
    # Standardized by the PC1 series' own full-record mean/std (not the
    # clim_start/clim_end window) — matches the original function exactly.
    pdo_index = _standardize(pc1_da) * -1

    pdo_index = _set_index_attrs(
        pdo_index, "PDO", "Pacific Decadal Oscillation",
        "Leading EOF (PC1) of area-weighted North Pacific SST anomalies, "
        "climatology and global mean removed.",
        units="std",
    )
    pdo_pattern = eof1.rename("pdo_pattern")

    return pdo_index, pdo_pattern


def calculate_pdo_ncl(sst, time_dim="time", lat_dim="lat", lon_dim="lon",
                       base_start="1920-01", base_end="2014-12"):
    """
    Compute the PDO index using the NCL reference methodology: the EOF is
    trained on `[base_start, base_end]` only, then the full record is
    projected onto that fixed spatial pattern (unlike `compute_pdo`,
    which re-computes the EOF over the whole record).

    Parameters
    ----------
    sst : xr.DataArray
        SST, any longitude convention (converted to 0-360 internally).
    time_dim, lat_dim, lon_dim : str
        Dimension names.
    base_start, base_end : str
        Reference period for the climatology, EOF training, and the PC1
        std used to scale the projected index.

    Returns
    -------
    pdo_index : xr.DataArray
        PDO index, standardized by the base period's PC1 std.
    eof1 : xr.DataArray
        Leading EOF spatial pattern trained on the base period.

    Notes
    -----
    This keeps the original NCL-translated math exactly as written
    (including projecting the *unweighted* full-record anomalies onto the
    *weighted*-space base-period eigenvector) so it stays numerically
    identical to the reference NCL script. Only the surrounding
    plumbing (longitude standardization, climatology, docstring, attrs)
    was restyled to match the rest of this module.
    """
    if sst[lon_dim].min() < 0:
        sst = sst.assign_coords({lon_dim: (sst[lon_dim] % 360)}).sortby(lon_dim)

    anomalies = _monthly_anomalies(sst, base_start, base_end, dim=time_dim)

    global_mean = _region_mean(anomalies, (-60, 70), (0, 360), lat_dim=lat_dim, lon_dim=lon_dim)
    anomalies = anomalies - global_mean

    pdo_full = _region_slice(anomalies, (20, 70), (110, 260), lat_dim=lat_dim, lon_dim=lon_dim)
    # Mask points invalid over the entire record (drifting sea ice etc.)
    pdo_full = pdo_full.where(pdo_full.notnull().all(dim=time_dim))
    pdo_base = pdo_full.sel({time_dim: slice(base_start, base_end)})

    eof_wgt = np.sqrt(np.cos(np.deg2rad(pdo_base[lat_dim].values)))
    nt, nlat, nlon = pdo_base.sizes[time_dim], pdo_base.sizes[lat_dim], pdo_base.sizes[lon_dim]
    Xb = pdo_base.fillna(0.0).values.reshape(nt, nlat * nlon)
    Xb = Xb * (eof_wgt[:, None] * np.ones((1, nlon))).reshape(-1)

    U, s, Vt = np.linalg.svd(Xb, full_matrices=False)
    eof_vec = Vt[0]
    explained_variance = float(s[0] ** 2 / (s ** 2).sum())

    nt_full = pdo_full.sizes[time_dim]
    Xf = pdo_full.fillna(0.0).values.reshape(nt_full, nlat * nlon)
    pc1 = Xf @ eof_vec

    eof_da = xr.DataArray(eof_vec.reshape(nlat, nlon), dims=[lat_dim, lon_dim],
                           coords={lat_dim: pdo_base[lat_dim], lon_dim: pdo_base[lon_dim]})
    eof_da, pc1 = _flip_sign_at_point(eof_da, pc1, lat_pt=37, lon_pt=200,
                                       lat_dim=lat_dim, lon_dim=lon_dim, positive=False)

    nt_base = pdo_base.sizes[time_dim]
    pc1 = pc1 / pc1[:nt_base].std()

    pdo_index = xr.DataArray(pc1, dims=[time_dim], coords={time_dim: pdo_full[time_dim]})
    pdo_index = _set_index_attrs(
        pdo_index, "PDO_NCL", "PDO index (NCL methodology)",
        "PC1 of the base-period EOF, projected onto the full record and "
        "standardized by the base-period PC std.",
        units="std", base_period=f"{base_start} to {base_end}",
        pc_variance_pct=explained_variance * 100,
    )
    return pdo_index, eof_da.rename("pdo_pattern")


def _polar_eof_index(da, pole_lat, lat_bounds, climatology_period, index_name, long_name):
    """
    Shared engine for the polar annular-mode indices (AAO and AO): the
    two original functions were identical apart from which pole/hemisphere
    and variable names they used, so that logic lives here once.

    Leading EOF/PC1 of area-weighted geopotential-height anomalies over
    `lat_bounds`, sign-corrected so index values are positive when the
    height anomaly at `pole_lat` is negative, standardized by the
    climatology period's PC1 std.
    """
    start, end = climatology_period
    region = da.sel(lat=_lat_slice(da, *lat_bounds))

    anomalies = _monthly_anomalies(region, start, end)
    weights = np.sqrt(np.cos(np.deg2rad(anomalies.lat)))
    weighted_anom = anomalies * weights

    solver = Eof(weighted_anom.values)
    eof1 = solver.eofs(neofs=1)
    pc1 = solver.pcs(npcs=1).flatten()

    pole_lat_idx = np.abs(region.lat - pole_lat).argmin().item()
    if eof1[0, pole_lat_idx, 0] > 0:
        pc1 = -pc1

    pc1_da = xr.DataArray(pc1, coords={"time": region.time}, dims=["time"])
    # Scaled (not demeaned) by the base period's std — matches the
    # original AAO/AO functions, which never subtract a mean here.
    index = _scale_by_std(pc1_da, start, end)

    return _set_index_attrs(
        index, index_name, long_name,
        f"EOF1 PC of monthly geopotential height anomalies poleward of "
        f"20 deg, sign-corrected at the pole.",
        units="std",
        climatology_period=f"{start}-{end}",
        normalization_period=f"{start}-{end}",
    )


def compute_aao(da_h700, climatology_period=("1979", "2000")):
    """
    Compute the Antarctic Oscillation (AAO) index (CPC methodology):
    leading EOF/PC1 of 700-hPa geopotential height anomalies poleward of
    20S.

    Parameters
    ----------
    da_h700 : xr.DataArray
        Monthly mean 700-hPa geopotential height, dims (time, lat, lon).
    climatology_period : tuple of str
        (start, end) years for climatology and normalization.

    Returns
    -------
    xr.DataArray
        Standardized monthly AAO index.
    """
    return _polar_eof_index(da_h700, pole_lat=-90, lat_bounds=(-90, -20),
                             climatology_period=climatology_period,
                             index_name="AAO", long_name="Antarctic Oscillation Index")


def compute_ao(da_h1000, climatology_period=("1978", "2000")):
    """
    Compute the Arctic Oscillation (AO) index (CPC methodology): leading
    EOF/PC1 of 1000-hPa geopotential height anomalies poleward of 20N.

    Parameters
    ----------
    da_h1000 : xr.DataArray
        Monthly mean 1000-hPa geopotential height, dims (time, lat, lon).
    climatology_period : tuple of str
        (start, end) years for climatology and normalization.

    Returns
    -------
    xr.DataArray
        Standardized monthly AO index.
    """
    return _polar_eof_index(da_h1000, pole_lat=90, lat_bounds=(20, 90),
                             climatology_period=climatology_period,
                             index_name="AO", long_name="Arctic Oscillation Index")


def compute_nao(slp, lat_bounds=(20, 80), lon_bounds=(-90, 40), time_mean="monthly",
                 clim_start=None, clim_end=None):
    """
    Compute the North Atlantic Oscillation (NAO) index as the leading EOF
    of sea-level-pressure (SLP) anomalies over the Atlantic sector
    (20-80N, 90W-40E). Positive NAO = a stronger-than-average pressure
    gradient between the Icelandic Low and Azores High.

    Parameters
    ----------
    slp : xr.DataArray
        Sea-level pressure, dims (time, lat, lon). lon in [-180, 180] or
        [0, 360]; units may be Pa or hPa (unit-agnostic).
    lat_bounds, lon_bounds : tuple of float
        Atlantic sector box. Default: 20-80N, 90W-40E.
    time_mean : {"monthly", "annual", "none"}, default "monthly"
        How anomalies are computed: monthly climatology removal
        (recommended), annual-mean removal only, or no anomaly at all.
    clim_start, clim_end : str or None
        Climatology reference period (e.g. "1991"/"2020"). None on both
        sides uses the full time extent as the reference.

    Returns
    -------
    xr.DataArray
        Standardized NAO index time series (dimensionless).

    Notes
    -----
    The original function's docstring also promised `eof1` and
    `explained_variance` as additional return values, but the code only
    ever returned `nao_index`. This version's docstring now matches the
    actual (single-value) return so it isn't misleading; the return
    value itself is unchanged from the original, so nothing that calls
    `compute_nao()` needs to change.
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    if float(slp.lon.max()) > 180:
        lon_min, lon_max = lon_min % 360, lon_max % 360

    slp_atl = slp.sortby("lat").sortby("lon").sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    if slp_atl.sizes["lat"] == 0 or slp_atl.sizes["lon"] == 0:
        raise ValueError(
            f"Atlantic sector subset is empty. Check lon convention. "
            f"lon range in data: [{float(slp.lon.min()):.1f}, {float(slp.lon.max()):.1f}]"
        )

    if time_mean == "monthly":
        slp_anom = _monthly_anomalies(slp_atl, clim_start, clim_end)
    elif time_mean == "annual":
        clim = slp_atl if (clim_start is None and clim_end is None) else slp_atl.sel(time=slice(clim_start, clim_end))
        slp_anom = slp_atl - clim.mean("time")
    elif time_mean == "none":
        slp_anom = slp_atl
    else:
        raise ValueError(f"time_mean must be 'monthly', 'annual', or 'none'. Got: '{time_mean}'")

    weights = np.sqrt(np.cos(np.deg2rad(slp_anom.lat)))
    slp_weighted = slp_anom * weights

    nt = slp_weighted.sizes["time"]
    slp_2d = np.asarray(slp_weighted).reshape(nt, -1)
    valid_mask = ~np.isnan(slp_2d).any(axis=0)
    slp_2d_valid = slp_2d[:, valid_mask]
    if slp_2d_valid.shape[1] == 0:
        raise ValueError("All spatial grid points contain NaN. Check your SLP data.")

    slp_2d_centered = slp_2d_valid - slp_2d_valid.mean(axis=0)
    U, S, Vt = np.linalg.svd(slp_2d_centered, full_matrices=False)
    eof1_valid = Vt[0, :]
    pc1 = U[:, 0] * S[0]
    explained_variance = float((S[0] ** 2) / np.sum(S ** 2))

    nlat, nlon = slp_weighted.sizes["lat"], slp_weighted.sizes["lon"]
    eof1_full = np.full(nlat * nlon, np.nan)
    eof1_full[valid_mask] = eof1_valid
    eof1_full = eof1_full.reshape(nlat, nlon)

    # Sign convention: positive loading over the Azores High (35-45N, 20W-0E)
    lats, lons = slp_anom.lat.values, slp_anom.lon.values
    azores_lat_mask = (lats >= 35) & (lats <= 45)
    azores_lon_mask = (lons >= 340) | (lons <= 360) if lons.max() > 180 else (lons >= -20) & (lons <= 0)
    azores_mean = np.nanmean(eof1_full[np.ix_(azores_lat_mask, azores_lon_mask)])
    if azores_mean < 0:
        eof1_full, pc1 = -eof1_full, -pc1

    nao_index_values = (pc1 - pc1.mean()) / pc1.std()
    clim_range_str = (f"{clim_start or str(slp_atl.time.values[0])[:10]} to "
                       f"{clim_end or str(slp_atl.time.values[-1])[:10]}")

    nao_index = xr.DataArray(nao_index_values, coords={"time": slp_anom.time}, dims=["time"])
    nao_index = _set_index_attrs(
        nao_index, "NAO", "North Atlantic Oscillation Index (PC-based)",
        "Standardized PC1 of SLP anomalies, 20-80N 90W-40E",
        units="std",
        sign_convention="Positive = stronger Azores High / Icelandic Low gradient",
        explained_variance_fraction=explained_variance,
        lat_bounds=str(lat_bounds), lon_bounds=str(lon_bounds),
        anomaly_method=time_mean, climatology_period=clim_range_str,
    )
    return nao_index


# ============================================================================
# Non-EOF pressure indices: SAM, NPI
# ============================================================================

def compute_sam(ds, var_name="prmsl", clim_start=1981, clim_end=2010):
    """
    Compute the Southern Annular Mode (SAM) index: the standardized
    pressure difference between 40S and 65S zonal-mean sea-level
    pressure (Python port of the NCL SAM index script).

    Parameters
    ----------
    ds : xr.Dataset
        Must contain `var_name` with dims (time, lat, lon).
    var_name : str, default "prmsl"
        Name of the sea-level-pressure variable.
    clim_start, clim_end : int
        Years bounding the base period used to standardize each
        calendar month.

    Returns
    -------
    xr.DataArray
        The SAM index time series.
    """
    zonal_40 = ds[var_name].sel(lat=-40, method="nearest").mean(dim="lon")
    zonal_65 = ds[var_name].sel(lat=-65, method="nearest").mean(dim="lon")

    norm_40 = _standardize_by_month(zonal_40, clim_start, clim_end)
    norm_65 = _standardize_by_month(zonal_65, clim_start, clim_end)

    index = norm_40 - norm_65
    if "month" in index.coords:
        index = index.drop_vars("month")

    return _set_index_attrs(
        index, "SAM", "Southern Annular Mode Index",
        "Standardized zonal-mean SLP difference between 40S and 65S "
        "(per-month standardization over the base period).",
        units="std",
        climatology_period=f"{clim_start}-{clim_end}",
    )


def compute_npi(slp, lat=None, lon=None, fill_value=None,
                 lat_bounds=(30.0, 65.0), lon_bounds=(160.0, 220.0)):
    """
    Compute the North Pacific Index (NPI): area-weighted mean sea-level
    pressure over 30N-65N, 160E-140W (Trenberth & Hurrell 1994).

    Unlike the other functions here, NPI works directly with numpy
    arrays internally so it accepts either an `xr.DataArray` or plain
    `slp`/`lat`/`lon` arrays.

    Parameters
    ----------
    slp : xr.DataArray or array_like
        Sea-level pressure. If an `xr.DataArray`, `lat`/`lon` are read
        from its coordinates and the result carries a time coordinate.
    lat, lon : array_like, optional
        Required if `slp` is not an `xr.DataArray`.
    fill_value : float, optional
        Value in `slp` to treat as missing (converted to NaN).
    lat_bounds, lon_bounds : tuple of float
        NPI box in degrees. `lon_bounds` is in 0-360 (default:
        30-65N, 160-220E, i.e. 160E-140W).

    Returns
    -------
    xr.DataArray
        NPI time series (or a scalar DataArray for a single time slice).
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    da_slp = slp if isinstance(slp, xr.DataArray) else None
    if da_slp is not None:
        lat = slp.lat.values
        lon = slp.lon.values
        slp = slp.values

    slp = np.asarray(slp, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    single_time = slp.ndim == 2
    if single_time:
        slp = slp[np.newaxis]

    lon360 = np.where(lon < 0, lon + 360.0, lon)
    lat_idx = (lat >= lat_min) & (lat <= lat_max)
    lon_idx = ((lon360 >= lon_min) & (lon360 <= lon_max) if lon_min <= lon_max
               else (lon360 >= lon_min) | (lon360 <= lon_max))

    slp_region = slp[:, lat_idx, :][:, :, lon_idx]
    if fill_value is not None:
        slp_region = np.where(slp_region == fill_value, np.nan, slp_region)

    w = np.cos(np.deg2rad(lat[lat_idx]))[np.newaxis, :, np.newaxis]
    weighted_sum = np.where(np.isnan(slp_region), 0.0, w * slp_region).sum(axis=(1, 2))
    weight_total = np.where(np.isnan(slp_region), 0.0, np.broadcast_to(w, slp_region.shape)).sum(axis=(1, 2))
    npi = weighted_sum / weight_total

    units = da_slp.attrs.get("units", "") if da_slp is not None else ""
    if single_time:
        return xr.DataArray(float(npi[0]), name="NPI",
                             attrs={"long_name": "North Pacific Index", "units": units})

    return xr.DataArray(
        npi, name="NPI",
        coords={"time": da_slp["time"]} if da_slp is not None else {},
        dims=["time"],
        attrs={
            "long_name": "North Pacific Index",
            "units": units,
            "description": (
                "Area-weighted mean SLP over 30N-65N, 160E-140W "
                "(Trenberth & Hurrell 1994)."
            ),
        },
    )
