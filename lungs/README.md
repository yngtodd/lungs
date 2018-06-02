## Profiling steps

###
`summit_main.py` is the file which uses 1K by 1K image
`1hj_main.py` : uses 256 by 256 image and code from HJ which achieved 40TF on Keras + TF

## Generate log file using nvprof

nvprof --metrics flop_count_sp,flop_count_hp --log-file half_log_2attempt.log python summmit_main.py

## calculating total number of flops

grep flop_count_sp half_log_2attempt.log | awk '{s+=$1*$9} END {printf "%.0f,", s}' >> count_2attempt.csv

## calculating number of FLOPS

run the progam again without nvrpof, time it and divide the number in csv file with the run time. thats the FLOPS
