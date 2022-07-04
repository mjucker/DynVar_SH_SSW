
source ~/.bashrc
module load nco

conda activate $data/envs/py3.9

declare -A mean_dict
levels=(10 20 30 50 70 100 250 300 500 700 850)
means=(30.0 25.0 23.0 20.0 18.0 15.0 9.5 8.5 5.0 3.5 1.0)
length=${#levels[@]}
for (( j=0; j<${length}; j++ ))
do
    mean_dict["${levels[$j]}"]=${means[$j]}
done

for file in $shared/ERA5/ERA5_dm.*.z.nc
do
    for level in ${levels[@]}
    do
	for delta_edge in -2.0 -1.5 -1.0 -0.5 +0.0 +0.5 +1.0 +1.5 +2.0
	do
	    edge=$(echo "${mean_dict[$level]}$delta_edge" |bc -l)
	    echo "$file; ${level}hPa, ${edge}km"
            ncap2 -s 'z=z/9.81' $file -O tmp.nc
	    pre=${file%.z.nc}
	    year=${pre#*.}
	    outFile=ERA5_vxmoms_${year}_${level}hPa_${edge}km.nc
	    python $repdir/DynVar_SH_SSW/compute_vortex_moments.py -z tmp.nc -l $level -e $edge -o $outFile
	done
    done
done
