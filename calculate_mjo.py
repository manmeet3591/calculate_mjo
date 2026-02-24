import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

# ----------------------------
# Utilities
# ----------------------------

def cosine_lat_weights(lat):
    """Cosine(lat) weights normalized to mean 1."""
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()

def area_mean_lat(ds, lat_name="lat"):
    """Latitude-weighted mean over latitude (returns lon x time)."""
    w = xr.DataArray(cosine_lat_weights(ds[lat_name].values), dims=[lat_name], coords={lat_name: ds[lat_name]})
    return (ds * w).mean(lat_name)

def dayofyear_climatology(da):
    """Daily climatology by dayofyear (handles leap days by grouping dayofyear)."""
    return da.groupby("time.dayofyear").mean("time")

def remove_daily_climatology(da):
    """Remove daily climatology (seasonal cycle) from a daily time series."""
    clim = dayofyear_climatology(da)
    return da.groupby("time.dayofyear") - clim

def remove_first_n_harmonics(da, n_harm=3):
    """
    Remove first n harmonics of annual cycle from daily data.
    This approximates Wheeler-Hendon style harmonic removal.
    """
    # time in days since start
    t = (da["time"] - da["time"][0]).astype("timedelta64[D]").astype(int)
    t = xr.DataArray(t, dims=["time"], coords={"time": da["time"]})
    year = 365.2425
    X = [xr.ones_like(t)]
    for k in range(1, n_harm + 1):
        X.append(np.cos(2 * np.pi * k * t / year))
        X.append(np.sin(2 * np.pi * k * t / year))
    X = xr.concat(X, dim="reg").transpose("time", "reg")  # (time, reg)

    # least squares regression per gridpoint (time, space)
    # Flatten spatial dims
    y = da.stack(space=[d for d in da.dims if d != "time"]).transpose("time", "space")
    # Solve beta = (X'X)^-1 X'y
    XtX = (X.T @ X).values
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T.values @ y.values)  # (reg, space)
    fit = (X.values @ beta)  # (time, space)

    resid = y - xr.DataArray(fit, dims=["time", "space"], coords={"time": da["time"], "space": y["space"]})
    return resid.unstack("space")

def standardize_over_time(da, eps=1e-12):
    """Standardize (z-score) using mean/std over time for each spatial point."""
    mu = da.mean("time")
    sd = da.std("time")
    return (da - mu) / (sd + eps), mu, sd

def to_feature_matrix(da_lon_time):
    """
    Convert (time, lon) DataArray into numpy (time, features).
    """
    # ensure dims are (time, lon)
    da2 = da_lon_time.transpose("time", ...)
    return da2.values

# ----------------------------
# Core RMM builder
# ----------------------------

def prepare_rmm_inputs(olr, u850, u200, lat_bounds=(-15, 15)):
    """
    olr/u850/u200: xarray DataArray with dims (time, lat, lon) and daily time coordinate.
    Returns combined feature matrix X (time, 3*lon) and coords.
    """
    # subset lat
    lat_name = "lat"
    olr = olr.sel({lat_name: slice(lat_bounds[0], lat_bounds[1])})
    u850 = u850.sel({lat_name: slice(lat_bounds[0], lat_bounds[1])})
    u200 = u200.sel({lat_name: slice(lat_bounds[0], lat_bounds[1])})

    # latitude-weighted mean to get (time, lon)
    olr_lon = area_mean_lat(olr, lat_name=lat_name)
    u850_lon = area_mean_lat(u850, lat_name=lat_name)
    u200_lon = area_mean_lat(u200, lat_name=lat_name)

    # remove mean seasonal cycle (daily climatology) then remove first 3 harmonics (often used)
    olr_anom = remove_first_n_harmonics(remove_daily_climatology(olr_lon), n_harm=3)
    u850_anom = remove_first_n_harmonics(remove_daily_climatology(u850_lon), n_harm=3)
    u200_anom = remove_first_n_harmonics(remove_daily_climatology(u200_lon), n_harm=3)

    # standardize each field over time (per lon)
    olr_std, olr_mu, olr_sd = standardize_over_time(olr_anom)
    u850_std, u850_mu, u850_sd = standardize_over_time(u850_anom)
    u200_std, u200_mu, u200_sd = standardize_over_time(u200_anom)

    # combine into one feature vector per day: [olr(lon...), u850(lon...), u200(lon...)]
    X = np.concatenate(
        [to_feature_matrix(olr_std), to_feature_matrix(u850_std), to_feature_matrix(u200_std)],
        axis=1
    )

    meta = {
        "lon": olr_lon["lon"].values,
        "time": olr_lon["time"].values,
        "standardization": {
            "olr": (olr_mu, olr_sd),
            "u850": (u850_mu, u850_sd),
            "u200": (u200_mu, u200_sd),
        }
    }
    return X, meta

def fit_rmm_eofs(X_train):
    """
    Fit the EOFs (PCA) on training matrix.
    Returns pca object; EOF patterns are pca.components_[0:2]
    """
    # Remove any remaining mean across time (PCA assumes centered)
    pca = PCA(n_components=2, whiten=False)
    pca.fit(X_train)
    return pca

def compute_rmm_from_pca(X, pca):
    """
    Project onto first 2 PCs -> RMM1, RMM2
    """
    pcs = pca.transform(X)  # (time, 2)
    rmm1 = pcs[:, 0]
    rmm2 = pcs[:, 1]
    amp = np.sqrt(rmm1**2 + rmm2**2)
    phase = rmm_phase(rmm1, rmm2)
    return rmm1, rmm2, amp, phase

def rmm_phase(rmm1, rmm2):
    """
    Convert RMM1/RMM2 to 1..8 phase (standard quadrant logic).
    Phase boundaries are 45-degree sectors starting from (rmm1>0,rmm2>0) etc.
    """
    ang = np.arctan2(rmm2, rmm1)  # -pi..pi
    ang_deg = (np.degrees(ang) + 360) % 360  # 0..360
    # 8 phases, each 45 degrees. Convention: phase 1 centered at 0 deg? varies.
    # Common WH convention uses:
    # Phase 1: 0-45? Often defined starting at 0 deg on +RMM1 axis and going CCW.
    # We'll define:
    # 1: 0-45, 2:45-90, ..., 8:315-360
    phase = (np.floor(ang_deg / 45).astype(int) + 1)
    phase[phase == 9] = 1
    return phase

# ----------------------------
# Example usage
# ----------------------------
# 1) Load your daily data (NetCDF) with xarray; variables must be DataArray with dims (time, lat, lon).
#
# ds = xr.open_dataset("yourfile.nc")
# olr  = ds["olr"]   # W/m^2 or similar
# u850 = ds["u850"]  # m/s
# u200 = ds["u200"]  # m/s
#
# 2) Choose training period (long is better; e.g., 20+ years). Here: select by time slice.
#
# olr_tr  = olr.sel(time=slice("1998-01-01", "2018-12-31"))
# u850_tr = u850.sel(time=slice("1998-01-01", "2018-12-31"))
# u200_tr = u200.sel(time=slice("1998-01-01", "2018-12-31"))
#
# X_train, meta_train = prepare_rmm_inputs(olr_tr, u850_tr, u200_tr)
# pca = fit_rmm_eofs(X_train)
#
# 3) Compute RMM for target period
#
# X, meta = prepare_rmm_inputs(olr, u850, u200)
# rmm1, rmm2, amp, phase = compute_rmm_from_pca(X, pca)
#
# 4) Put into an xarray Dataset
#
# out = xr.Dataset(
#     {
#         "RMM1": (("time",), rmm1),
#         "RMM2": (("time",), rmm2),
#         "amplitude": (("time",), amp),
#         "phase": (("time",), phase),
#     },
#     coords={"time": meta["time"]},
# )
# print(out)
