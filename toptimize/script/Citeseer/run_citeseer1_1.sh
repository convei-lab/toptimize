# Path
cd /data/brandon/toptimize/toptimize

### GCN ###

## Ours
# python train.py "Citeseer/no_drop_real" -b GCN -d Citeseer -tr 100 -t 7

## No LL
# python train.py "Citeseer/no_drop_LL_real" -b GCN -d Citeseer -tr 100 -t 9999999 -l1 0

## No DL
# python train.py "Citeseer/no_drop_DL_real" -b GCN -d Citeseer -tr 100 -t 7 -l2 0

### GAT ###

# OURs
# python train.py "Citeseer/no_drop_trial2_hs8" -b GAT -d Citeseer -tr 100 -t 0.6 -te 200
# python train.py "Citeseer/no_drop_LL_trial2_hs8" -b GAT -d Citeseer -tr 100 -t 99999999 -te 200 -l1 0
# python train.py "Citeseer/no_drop_DL_trial2_hs8" -b GAT -d Citeseer -tr 100 -t 0.6 -te 200 -l2 0

python train.py "Citeseer/no_DL_paper_0.4" -b GAT -d Citeseer -tr 100 -t 0.4 -l2 0

# No LL
# python train.py "Citeseer/no_drop_LL_trial1" -b GAT -d Citeseer -tr 100 -t 100000000000 -te 300 -l1 0

# No DL
# python train.py "Citeseer/no_drop_DL_trial1" -b GAT -d Citeseer -tr 100 -t 0.2 -te 300 -l2 0
### Cold Start ###

# GCN (0.25, 0.50, 0.75)
# python train.py "Coldstart/Citeseer/0.25_real" -b GCN -d Citeseer -tr 50 -t 7 -csr 0.25
# python train.py "Coldstart/Citeseer/0.50_real" -b GCN -d Citeseer -tr 50 -t 7 -csr 0.50
# python train.py "Coldstart/Citeseer/0.75_real" -b GCN -d Citeseer -tr 50 -t 7 -csr 0.75

# GAT (0.25, 0.50, 0.75)