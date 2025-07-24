#!/bin/bash

set -e  # optional: stop on first error

cd /mnt/d/Sailfish_new/ClusterTests/Test3/Output

for f in chkpt.*.pk; do
    python /mnt/d/Sailfish_new/sailfish/plot.py "$f" -o /mnt/d/Sailfish_new/ClusterTests/Test3/Figures
done
