#!/bin/bash
#python average_weights.py --summitdev --num_epochs 5 
python hpml.py --summitdev --parallel --num_epochs 50 --batch_size 4 
gdb --batch python core.* -ex "thread apply all bt"
