# Pilot Experiment
# python train.py seed0 -s 0 -b GCN -d Cora -r 20 -t 5 -l -w
# python train.py seed1 -s 1 -b GCN -d Cora -r 20 -t 5 -l -w
# python train.py seed2 -s 2 -b GCN -d Cora -r 20 -t 5 -l -w

# GCN

# Cora
# python train.py no_drop_lr0.005 -s 0 -b GCN -d Cora -r 20 -m 10

# Citeseer
# python train.py drop_alpha10 -s 0 -b GCN -d Citeseer -r 20 -m 10 -p
# python train.py drop_alpha9 -s 0 -b GCN -d Citeseer -r 20 -m 9 -p

# Cold Start
python train.py cs_0.75 -b GCN -d Citeseer -r 50 -m 7 -c 0.75


# Drop Experiment (Best Epoch)
# python train.py d_b0 -s 0 -b GCN -d Cora -r 100 -t 5 -dr
# python train.py d_b1 -s 1 -b GCN -d Cora -r 100 -t 5 -dr
# python train.py d_b2 -s 2 -b GCN -d Cora -r 100 -t 5 -dr

# Drop Experiment (Best Epoch)
# python train.py d_b0 -s 0 -b GCN -d Citeseer -r 20 -t 5 -dr -w
# python train.py d_b1 -s 1 -b GCN -d Citeseer -r 20 -t 5 -dr -w
# python train.py d_b2 -s 2 -b GCN -d Citeseer -r 20 -t 5 -dr -w

# Drop Experiment (Best Epoch)
# python train.py d_b0 -s 0 -b GCN -d Pubmed -r 20 -t 5 -dr -w
# python train.py d_b1 -s 1 -b GCN -d Pubmed -r 20 -t 5 -dr -w
# python train.py d_b2 -s 2 -b GCN -d Pubmed -r 20 -t 5 -dr -w

# Drop Experiment (Last Epoch)
# python train.py d_l0 -s 0 -b GCN -d Cora -r 20 -t 5 -dr -w -l
# python train.py d_l1 -s 1 -b GCN -d Cora -r 20 -t 5 -dr -w -l
# python train.py d_l2 -s 2 -b GCN -d Cora -r 20 -t 5 -dr -w -l

# Drop Experiment (Last Epoch)
# python train.py d_l0 -s 0 -b GCN -d Citeseer -r 20 -t 5 -dr -w -l
# python train.py d_l1 -s 1 -b GCN -d Citeseer -r 20 -t 5 -dr -w -l
# python train.py d_l2 -s 2 -b GCN -d Citeseer -r 20 -t 5 -dr -w -l

# Drop Experiment (Last Epoch)
# python train.py d_l0 -s 0 -b GCN -d Pubmed -r 20 -t 5 -dr -w -l
# python train.py d_l1 -s 1 -b GCN -d Pubmed -r 20 -t 5 -dr -w -l
# python train.py d_l2 -s 2 -b GCN -d Pubmed -r 20 -t 5 -dr -w -l

# # GAT
# python train.py no_drop_LL_real -b GAT -d Cora -r 100 -m 4 -l 0 -e 600

# Citeseer
# python train.py no_drop_LL_real -b GAT -d Citeseer -r 100 -m 10000 -e 500 -l 0

# # Drop Experiment (Best Epoch)
# python train.py d_b0 -s 0 -b GAT -d Cora -r 20 -t 5 -dr -w
# python train.py d_b1 -s 1 -b GAT -d Cora -r 20 -t 5 -dr -w
# python train.py d_b2 -s 2 -b GAT -d Cora -r 20 -t 5 -dr -w

# # Drop Experiment (Best Epoch)
# python train.py d_b0 -s 0 -b GAT -d Citeseer -r 20 -t 5 -dr -w
# python train.py d_b1 -s 1 -b GAT -d Citeseer -r 20 -t 5 -dr -w
# python train.py d_b2 -s 2 -b GAT -d Citeseer -r 20 -t 5 -dr -w

# # Drop Experiment (Best Epoch)
# python train.py d_b0 -s 0 -b GAT -d Pubmed -r 20 -t 5 -dr -w
# python train.py d_b1 -s 1 -b GAT -d Pubmed -r 20 -t 5 -dr -w
# python train.py d_b2 -s 2 -b GAT -d Pubmed -r 20 -t 5 -dr -w

# # Drop Experiment (Last Epoch)
# python train.py d_l0 -s 0 -b GAT -d Cora -r 20 -t 5 -dr -w -l
# python train.py d_l1 -s 1 -b GAT -d Cora -r 20 -t 5 -dr -w -l
# python train.py d_l2 -s 2 -b GAT -d Cora -r 20 -t 5 -dr -w -l

# # Drop Experiment (Last Epoch)
# python train.py d_l0 -s 0 -b GAT -d Citeseer -r 20 -t 5 -dr -w -l
# python train.py d_l1 -s 1 -b GAT -d Citeseer -r 20 -t 5 -dr -w -l
# python train.py d_l2 -s 2 -b GAT -d Citeseer -r 20 -t 5 -dr -w -l

# # Drop Experiment (Last Epoch)
# python train.py d_l0 -s 0 -b GAT -d Pubmed -r 20 -t 5 -dr -w -l
# python train.py d_l1 -s 1 -b GAT -d Pubmed -r 20 -t 5 -dr -w -l
# python train.py d_l2 -s 2 -b GAT -d Pubmed -r 20 -t 5 -dr -w -l

# Attack Experiment
# python attack_n_defense.py seed1 -s 1 -b GCN -d Cora -r 20 -t 5 -l -w