#!/usr/bin/bash
source ~/.bashrc
conda activate ag

# python ./Crab_createData.py
python ./Crab_ag.py
python ./Expt_cutData.py
python ./Expt_isgamma_E_Ra_Dec.py