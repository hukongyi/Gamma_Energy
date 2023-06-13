#!/usr/bin/bash
source /home2/hky/miniconda3/etc/profile.d/conda.sh
# conda info
doit() {
   x=$1
   echo $x
   conda activate healpy
   python ./getBackground.py $x
}
export -f doit

seq 0 39|parallel -j40 doit