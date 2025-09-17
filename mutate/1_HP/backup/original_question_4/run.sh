#!/bin/bash -l

OUT_0=./acc/1_original.out
ERR_0=./acc/1_original.err
python 1_original_rouge.py 1>>$OUT_0 2>>$ERR_0 &

python 2_single_mutation.py

python 3_evaluation\(single_gpt-claude\).py

OUT_1=./acc/2_mutation.out
ERR_1=./acc/2_mutation.err
python 4_cal_avg_rouge.py 1>>$OUT_1 2>>$ERR_1

python 6_single_mutation_with_few_shot_examples.py

python 7_evaluation\(few_shot_version\).py

OUT_2=./acc/3_few_shots.out
ERR_2=./acc/3_few_shots.err
python 8_cal_avg_rouge_with_few_shots 1>>$OUT_2 2>>$ERR_2
