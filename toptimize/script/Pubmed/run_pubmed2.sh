# Path
cd /data/brandon/toptimize/toptimize

### GCN ###

## Ours
# python train.py "Pubmed/no_drop_real" -b GCN -d Pubmed -tr 100 -t 9

## No LL
# python train.py "Pubmed/no_LL_paper" -b GCN -d Pubmed -tr 100 -t 9999999 -l1 0

## No DL
# python train.py "Pubmed/no_drop_DL_real" -b GCN -d Pubmed -tr 100 -t 9 -l2 0
# python train.py "Pubmed/Ours_paper_DL" -b GCN -d Pubmed -tr 100 -t 9 -l2 0
### GAT ###

#Ours
python train.py "dev-5" -b GAT -d Pubmed -tr 20 -t 5 -sm -ts 20

# No LL

# No DL

### Cold Start ###

# GCN (0.25, 0.50, 0.75)
# python train.py "Coldstart/Pubmed/0.25_real" -b GCN -d Pubmed -tr 50 -t 10 -csr 0.25
# python train_pubmed.py "Coldstart/Pubmed/0.50_real" -b GCN -d Pubmed -tr 100 -t 12 -csr 0.50
# python train.py "Coldstart/Pubmed/0.75_real" -b GCN -d Pubmed -tr 50 -t 10 -csr 0.75

# GAT (0.25, 0.50, 0.75)
# python train.py "Coldstart/Pubmed/0.25_paper" -b GAT -d Pubmed -tr 50 -t 5 -csr 0.25