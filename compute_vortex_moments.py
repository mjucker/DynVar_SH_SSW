import xarray as xr
from aostools import climate as ac
from aostools import inout as ai
from vortex_moments import vor
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-z',dest='z_file',help='File containing Z10.')
#parser.add_argument('-n',dest='label',help='label for file name.')
parser.add_argument('-Z',dest='z10',default=None,help='Name of Z10 variable. If None, it is assumed there is only one variable in z_file.')
parser.add_argument('-l',dest='level',default=None,type=float,help='Extract this pressure level.')
parser.add_argument('-e',dest='edge',default=30.2,type=float,help='Value of edge of polar vortex [km].')
parser.add_argument('-o',dest='outFile',help="Name of output file")
args = parser.parse_args()

if args.z10 is None:
    z = xr.open_dataarray(args.z_file)
else:
    z = xr.open_dataset(args.z_file)[args.z10]

z = ac.StandardGrid(z,rename=True)
if args.level is not None:
    z = z.sel(pres=args.level)

z10 = z.values

lons = z.lon.values
lats = z.lat.values
aspects = np.zeros(len(z.time),)
latc = np.zeros_like(aspects)
lonc=np.zeros_like(aspects)
nt=len(z.time)
for t in range(nt):
         ac.update_progress(t/nt)
         moms = vor.calc_moments(z10[t,:],lats,lons,hemisphere='SH',field_type='GPH',edge=args.edge*1000)
         aspects[t] = moms['aspect_ratio']
         latc[t] = moms['centroid_latitude']
         lonc[t] = moms['centroid_longitude']

aspx = xr.DataArray(aspects,coords=[z.time],name='aspect_ratio')
latx = xr.DataArray(latc,coords=[z.time],name='centroid_latitude')
lonx = xr.DataArray(lonc,coords=[z.time],name='centroid_longitude')

#outFile = 'results/vxmoms_composite_{0}.nc'.format(args.label)
outFile = args.outFile
xr.merge([aspx,latx,lonx]).to_netcdf(outFile)
print(outFile)
