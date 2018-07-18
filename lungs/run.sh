#!/bin/bash
#python average_weights.py --summitdev --num_epochs 5 
python hpml_no_hvd.py --summitdev --parallel --num_epochs 60 --batch_size 32 --checkpoint-format './training_scratch/checkpoint-{epoch}.pth.tar'
gdb --batch python core.* -ex "thread apply all bt"
