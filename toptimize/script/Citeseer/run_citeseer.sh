# Path
cd /data/brandon/toptimize/toptimize

### GCN ###

## Ours
# python train.py "Citeseer/no_drop_new_trial1" -b GCN -d Citeseer -tr 100 -t 9 -te 200 -et
# python train.py "Citeseer/no_drop_LL_new_trial1" -b GCN -d Citeseer -tr 100 -t 99999999 -te 200 -l1 0
# python train.py "Citeseer/no_drop_DL_new_trial1" -b GCN -d Citeseer -tr 100 -t 9 -te 200 -l2 0
python train.py "dev_6" -b GCN -d Citeseer -tr 20 -t 6 -te 300 -ts 20 -sm 
# python train.py "dev3" -b GCN -d Cora -tr 20 -t 8 -te 300 -ts 20 -sm

## No LL
# python train.py "Citeseer/no_drop_LL_real" -b GCN -d Citeseer -tr 100 -t 9999999 -l1 0

## No DL
# python train.py "Citeseer/no_drop_DL_real" -b GCN -d Citeseer -tr 100 -t 7 -l2 0

### GAT ###

## Ours
# python train.py "Citeseer/no_drop_test1" -b GAT -d Citeseer -tr 10 -t 1 -te 300 -ts 10
# python train.py "Citeseer/Ours_paper_0.3" -b GAT -d Citeseer -tr 100 -t 0.3
# python train.py "Citeseer/Ours_dropout" -b GAT -d Citeseer -tr 100 -ts 5 -t 0.2 -l1 1 -l2 10 -hs 8 # -ea #-wnb
## No LL

## No DL

### Cold Start ###

# GCN (0.25, 0.50, 0.75)
# python train.py "Coldstart/Citeseer/0.25_real" -b GCN -d Citeseer -tr 50 -t 7 -csr 0.25
# python train.py "Coldstart/Citeseer/0.50_real" -b GCN -d Citeseer -tr 50 -t 7 -csr 0.50
# python train.py "Coldstart/Citeseer/0.75_real" -b GCN -d Citeseer -tr 50 -t 7 -csr 0.75

# # GAT (0.25, 0.50, 0.75)
# python train.py "Coldstart/Citeseer/0.25_trial1" -b GAT -d Citeseer -tr 100 -t 0.2 -csr 0.25 -te 300
# python train.py "Coldstart/Citeseer/0.50_trial1" -b GAT -d Citeseer -tr 100 -t 0.2 -csr 0.50 -te 300
# python train.py "Coldstart/Citeseer/0.75_trial1" -b GAT -d Citeseer -tr 100 -t 0.2 -csr 0.75 -te 300