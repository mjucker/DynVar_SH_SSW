import xarray as xr
from aostools import climate as ac
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l',dest='levels',default=None,nargs='+',help='list of pressure levels [hPa] to analyse')
parser.add_argument('-e',dest='edges',default=None,nargs='+',help='list of polar vortex edge heights [km] to analyse.')
parser.add_argument('-s',dest='seasons',default=None,nargs='+',help='list of seasons to analyse.')
parser.add_argument('-x',dest='xlims',default=[0.8,3.2],nargs=2,type=float,help='set x-limits for plot (aspect ratio).')
parser.add_argument('-y',dest='ylims',default=[58,93],nargs=2,type=float,help='set y-limits for plot (central latitude).')
parser.add_argument('-m',dest='max',action='store_true',help='look for rolling mean max of aspect ration and min of centroid latitude.')
parser.add_argument('-r',dest='roll',default=7,help='rolling max/min for aspect ration in days. Only used if -m.')
args = parser.parse_args()


if args.levels is None:
    from glob import glob
    all_files = glob('vxmoms/*.nc')
    all_files.sort()
    levels = [int(i.split('hPa')[0].split('_')[-1]) for i in all_files]
    levels = np.unique(levels)
else:
    levels = args.levels
if args.seasons is None:
    seasons = ['MAM','JJA','SON']
else:
    seasons = args.seasons

sns.set_style('whitegrid')
sns.set_context('paper')
colors = sns.color_palette()

nedges = 0
dl = []
for l,level in enumerate(levels):
    dss = []
    for season in seasons:
        if args.edges is None:
            from glob import glob
            all_files = glob('vxmoms/*_{0}hPa_*.nc'.format(level))
            all_files.sort()
            edges  = [float(i.split('km')[0].split('_')[-1]) for i in all_files]
            edges  = np.unique(edges)
        else:
            edges = args.edges
        nedges = len(edges)
        de = []
        for edge in edges:
            inFiles = 'vxmoms/ERA5_vxmoms_????_{0}hPa_{1}km.nc'.format(level,edge)
            ds = xr.open_mfdataset(inFiles)
            if args.max:
                # aspect ratio: looking for above threshold
                ds['aspect_ratio'] = ds.aspect_ratio.rolling(time=args.roll).min()
                # centroid latitude: looking for below threshold
                ds['centroid_latitude'] = ds.centroid_latitude.rolling(time=args.roll).max()
            filtr = ds.time.dt.season == season
            ds = ds.isel(time=filtr)
            ds['edged'] = edge
            de.append(ds)
        ds = xr.concat(de,dim='edged')
        edgev = ds.edged.expand_dims(time=ds.time)
        edgev.name = 'edge'
        ds = xr.merge([ds,edgev.transpose('edged','time')])
        ds = ds.stack(time_edge=['time','edged'])
        fg = sns.FacetGrid(ds.to_dataframe(),col='edge',dropna=False,col_wrap=3,xlim=args.xlims,ylim=args.ylims)
        fg.map_dataframe(sns.kdeplot,x='aspect_ratio',y='centroid_latitude',color=colors[l])
        fg.map_dataframe(sns.scatterplot,x='aspect_ratio',y='centroid_latitude',color=colors[l],alpha=0.3)
        #for a,ax in enumerate(fg.axes.flatten()):
        #    txt = 'edge = {0}km'.format(edges[a])
        #    ax.text(1.0,1.0,txt,ha='right',va='top',transform=ax.transAxes)
        txt = 'level = {0}hPa; season = {1}'.format(level,season)
        fg.figure.suptitle(txt)
        outFile = 'figures/ERA5_vxmoms_{0}_{1}hPa.pdf'.format(season,level)
        if args.max:
            outFile = outFile.replace('.pdf','_r{0}d.pdf'.format(args.roll))
        fg.savefig(outFile,transparent=True)
        print(outFile)
        plt.close()
        ds['season'] = season
        dss.append(ds)
    dss = xr.concat(dss,dim='season')
    dss['level'] = level
    dl.append(dss)
dl = xr.concat(dl,dim='level')

# plot 1D KDEs

def PlotKDE(da,var,lims,ax,fill=False):
    plotArgs = {
                'x'   : var,
                'data': da,
                'hue' : 'edge',
                'common_norm':False,
                'clip': lims,
                'common_grid':True,
                'palette': 'crest',
                'ax'  : ax,
                'fill': fill
               }
    if fill:
        plotArgs['alpha'] = 0.5
        plotArgs['linewidth'] = 0
    p = sns.kdeplot(**plotArgs)
    handles = p.legend_.legendHandles[::-1]
    #if fill:
        # unfortunately, p.collections (instead of p.lines) does not work with dashes, only with linestyles
        #for line,ls,handle in zip(p.collections,dashes,handles):
        #    line.set_dashes(ls)
        #    handle.set_dashes(ls)
    if not fill:
        for line,ls,handle in zip(p.lines,dashes,handles):
            line.set_dashes(ls)
            handle.set_dashes(ls)
    ax.set_xlim(lims)
    ylims = ax.get_ylim()
    if args.max:
        ax.set_ylim(0,min(10,ylims[-1]))
    else:
        ax.set_ylim(0,min(3,ylims[-1]))
    

# dl is now a function of level, season, and time_edge
nlevs = len(dl.level)
if nlevs == 4:
    ncols = 2
elif nlevs > 4:
    ncols = 3
else:
    ncols = nlevs
nrows = 1+(nlevs-1)//ncols
# get a collection of linestyles
dashes = sns._core.unique_dashes(nedges)
# set the middle edge (closest to climatology) to solid line
dashtmp = dashes[nedges//2]
dashes[nedges//2] = dashes[0]
dashes[0] = dashtmp
for season in dl.season:
    figa,axa = plt.subplots(nrows=nrows,ncols=ncols,figsize=[4*ncols,3*nrows],sharex=True,sharey=True)
    figl,axl = plt.subplots(nrows=nrows,ncols=ncols,figsize=[4*ncols,3*nrows],sharex=True,sharey=True)
    for l,level in enumerate(dl.level):
        if nrows > 1:
            n = l//nrows
            m = l-n*nrows
            ax = axa[n][m]
        else:
            ax = axa[l]
        dltmp = dl.sel(season=season,level=level)
        # filled transparent kdes or dashed lines
        PlotKDE(dltmp,'aspect_ratio',args.xlims,ax,fill=False)
        ax.set_title('{0}hPa'.format(level.values))
        if nrows > 1:
            ax = axl[n][m]
        else:
            ax = axl[l]
        PlotKDE(dltmp,'centroid_latitude',args.ylims,ax,fill=False)
        ax.set_title('{0}hPa'.format(level.values))
    figa.suptitle('aspect ratio, {}'.format(season.values))
    figl.suptitle('centroid latitude, {}'.format(season.values))
    outFile = 'figures/ERA5_aspect_ratio_{0}.pdf'.format(season.values)
    if args.max:
        outFile = outFile.replace('.pdf','_r{0}d.pdf'.format(args.roll))
    figa.savefig(outFile,transparent=True)
    print(outFile)
    outFile = outFile.replace('aspect_ratio','centroid_latitude')
    figl.savefig(outFile,transparent=True)
    print(outFile)


        


        

            
