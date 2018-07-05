#!/bin/bash
#python average_weights.py --summitdev --num_epochs 5 
python hpml.py --parallel --num_epochs 60 --batch_size 32 --checkpoint-format './node16/checkpoint-{epoch}.pth.tar'
#gdb --batch python core.* -ex "thread apply all bt"
