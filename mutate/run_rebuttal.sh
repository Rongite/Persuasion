#!/bin/bash

python 1_run_rebuttal.py --book 2_HB --technique "Ethos" --technique_dir 1_Ethos
python 2_inference_scaling_all.py --book 2_HB --technique "Ethos" --technique_dir 1_Ethos
# python 3_rebuttal_statistics.py --book 2_HB --technique "Ethos" --technique_dir 1_Ethos
