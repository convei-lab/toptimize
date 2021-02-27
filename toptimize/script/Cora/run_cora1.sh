# Path
cd /data/brandon/toptimize/toptimize

##### GCN #####

## Ours
# python train.py "Cora/no_drop_real" -b GCN -d Cora -tr 100 -t 10

## No LL
# python train.py "Cora/no_drop_LL_real" -b GCN -d Cora -tr 100 -t 9999999 -l1 0

## No DL
# python train.py "Cora/no_drop_DL_real" -b GCN -d Cora -tr 100 -t 10 -l2 0

##### GAT #####

## Ours
# On Test
# python train.py "Cora/no_drop_test3" -b GAT -d Cora -tr 10 -t 3 -te 300
python train.py "Cora/no_drop_test9" -b GAT -d Cora -tr 10 -t 9 -te 300

## No LL

## No DL

##### Cold Start #####

## GCN (0.25, 0.50, 0.75)
# python train.py "Coldstart/Cora/0.25_real" -b GCN -d Cora -tr 50 -t 10 -csr 0.25
# python train.py "Coldstart/Cora/0.50_real" -b GCN -d Cora -tr 50 -t 10 -csr 0.50
# python train.py "Coldstart/Cora/0.75_real" -b GCN -d Cora -tr 50 -t 10 -csr 0.75

## GAT (0.25, 0.50, 0.75)