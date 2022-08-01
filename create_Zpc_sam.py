import xarray as xr
from aostools import climate as ac
from dask.diagnostics import ProgressBar

data_dir = '/srv/ccrc/AtmMJ/shared/ERA5/'


clim = ['1981','2010']

z = xr.open_mfdataset(data_dir+'ERA5_dm.*.z.nc')
# convert geopotential to geopotential height
z = z.z/9.81
z = ac.StandardGrid(z,rename=True)

# polar cap average
z = ac.GlobalAvgXr(z,[-90,-60]).mean('lon')
    

z_clim = z.sel(time=slice(*clim)).groupby('time.dayofyear').mean()

za = z.groupby('time.dayofyear') - z_clim

# SAM has inverse sign to polar cap Z anomaly!
zs = za.groupby('time.dayofyear')/za.groupby('time.dayofyear').std()
zs = -zs

delayed = zs.to_netcdf('zpc_sam/zpc_sam.nc',compute=False)
with ProgressBar():
    delayed.compute()


