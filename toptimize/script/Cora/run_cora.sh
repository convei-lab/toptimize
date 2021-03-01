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
# python train.py "Cora/ours_paper_LL_2trial" -b GAT -d Cora -tr 100 -t 9999999999 -te 300 -l1 0
# python train.py "Cora/no_drop_trial2" -b GAT -d Cora -tr 100 -t 4 -te 200 -hs 8
# python train.py "Cora/no_drop_LL_trial2" -b GAT -d Cora -tr 100 -t 9999999999 -te 200 -hs 8 -l1 0
# python train.py "Cora/no_drop_DL_trial2" -b GAT -d Cora -tr 100 -t 4 -te 200 -hs 8 -l2 0
# python train.py "Cora/test_selftraining" -b GCN -d Cora -tr 50 -t 10 -te 300 -et
# python train.py "Cora/no_drop_test1000" -b GAT -d Cora -tr 10 -t 4 -te 1000
# python train.py "Cora/no_drop_sofar2" -b GAT -d Cora -tr 100 -t 10 -te 300

## No LL

## No DL

##### Cold Start #####

## GCN (0.25, 0.50, 0.75)
python train.py "Coldstart/Cora/0_paper" -b GCN -d Cora -tr 100 -t 8 -csr 0.0005
# python train.py "Coldstart/Cora/0.25_real" -b GAT -d Cora -tr 100 -t 10 -csr 0.25 -te 300
# python train.py "Coldstart/Cora/0.25_real" -b GAT -d Cora -tr 100 -t 10 -csr 0.25 -te 300
# python train.py "Coldstart/Cora/0.50_real" -b GAT -d Cora -tr 100 -t 10 -csr 0.50 -te 300
# python train.py "Coldstart/Cora/0.75_real" -b GAT -d Cora -tr 100 -t 10 -csr 0.75 -te 300

## GAT (0.25, 0.50, 0.75)