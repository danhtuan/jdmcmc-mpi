#!/bin/bash
echo "/usr/lib64/openmpi/bin/mpirun --mca btl_tcp_if_exclude virbr0 -H moria,shire,gondor -n 36 ./gibbs -f images/discs8.bmp -n 8 -i 100 -g 1000"
/usr/lib64/openmpi/bin/mpirun --mca btl_tcp_if_exclude virbr0 -H moria,shire,gondor -n 36 ./gibbs -f images/discs8.bmp -n 8 -i 100 -g 1000
