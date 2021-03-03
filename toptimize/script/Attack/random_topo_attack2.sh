# Adversarial attack with different edge perturbation ratios
# python pgd_attack.py dev ptb5 -tr 1 -ts 5 -ptb 0.05
# python pgd_attack.py dev ptb10 -tr 1 -ts 5 -ptb 0.10
# python pgd_attack.py dev ptb15 -tr 1 -ts 5 -ptb 0.15
# python pgd_attack.py dev ptb20 -tr 1 -ts 5 -ptb 0.20

# python attack.py ptb5 pgd_attack dev_Cora_GCN -vr 0 -vm 5 -vt 0 -ptb 0.05

# Pilot
# python attack.py pilot pgd_attack dev-7_Cora_GCN -vr 20 -vm 0 -vt 20 -ptb 0.05

# Required params
# att_alias 
# victim_name 
# attack_type

# Important params
# victim_model_run 
# victim_model_step 
# victim_topo_step
# ptb

# Example robust topology experiment
# PA-GCN ptb 5
# PA-GCN(X, A0') <- GCN(X, A0)
# PA-GCN(X, A20') <- GCN(X, A20)
# python attack.py base-ptb5 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0 -ptb 0.05
# python attack.py ours-ptb5 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.05

# robust topology experiment CORA
# python attack.py "topo-big2/base-ptb5" random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.5 -ca -ts 0
# python attack.py "topo-big2/ours-ptb5" random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.5 -ca -ts 0
# python attack.py "topo-big2/base-ptb10" random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 1.0 -ca -ts 0
# python attack.py "topo-big2/ours-ptb10" random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 1.0 -ca -ts 0
# python attack.py "topo-big2/base-ptb15"  random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 1.5 -ca -ts 0
# python attack.py "topo-big2/ours-ptb15"  random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 1.5 -ca -ts 0
# python attack.py "topo-big2/base-ptb20"  random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 2.0 -ca -ts 0
# python attack.py "topo-big2/ours-ptb20"  random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 2.0 -ca -ts 0

# python attack.py "temp/base-ptb5"  random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.05 -ca -ts 0
# python attack.py "temp/ours-ptb5"  random_attack dev-8_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.05 -ca -ts 0

# robust topology experiment Citeseer
# python attack.py "topo-big2/base-cite-ptb5" random_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 0  -ptb 0.5 -ca -ts 0
# python attack.py "topo-big2/ours-cite-ptb5" random_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 20 -ptb 0.5 -ca -ts 0
# python attack.py "topo-big2/base-cite-ptb10" random_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 0  -ptb 1.0 -ca -ts 0
# python attack.py "topo-big2/ours-cite-ptb10" random_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 20 -ptb 1.0 -ca -ts 0
# python attack.py "topo-big2/base-cite-ptb15"  random_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 0  -ptb 1.5 -ca -ts 0
# python attack.py "topo-big2/ours-cite-ptb15"  random_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 20 -ptb 1.5 -ca -ts 0
# python attack.py "topo-big2/base-cite-ptb20"  random_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 0  -ptb 2.0 -ca -ts 0
# python attack.py "topo-big2/ours-cite-ptb20"  random_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 20 -ptb 2.0 -ca -ts 0

# robust topology experiment CORA
# python attack.py "topo-big2/base-GAT-ptb5" random_attack dev-8_Cora_GAT -tr 19 -vm 0 -vt 0  -ptb 0.5 -ca -ts 0
# python attack.py "topo-big2/ours-GAT-ptb5" random_attack dev-8_Cora_GAT -tr 19 -vm 0 -vt 20 -ptb 0.5 -ca -ts 0
python attack.py "topo-big2/base-GAT-ptb10" random_attack dev-8_Cora_GAT -tr 19 -vm 0 -vt 0  -ptb 1.0 -ca -ts 0
# python attack.py "topo-big2/ours-GAT-ptb10" random_attack dev-8_Cora_GAT -tr 19 -vm 0 -vt 20 -ptb 1.0 -ca -ts 0
# python attack.py "topo-big2/base-GAT-ptb15"  random_attack dev-8_Cora_GAT -tr 19 -vm 0 -vt 0  -ptb 1.5 -ca -ts 0
# python attack.py "topo-big2/ours-GAT-ptb15"  random_attack dev-8_Cora_GAT -tr 19 -vm 0 -vt 20 -ptb 1.5 -ca -ts 0
# python attack.py "topo-big2/base-GAT-ptb20"  random_attack dev-8_Cora_GAT -tr 19 -vm 0 -vt 0  -ptb 2.0 -ca -ts 0
# python attack.py "topo-big2/ours-GAT-ptb20"  random_attack dev-8_Cora_GAT -tr 19 -vm 0 -vt 20 -ptb 2.0 -ca -ts 0


# robust topology experiment Citeseer
# python attack.py "topo-big2/base-GAT-cite-ptb5" random_attack dev-7_Citeseer_GAT -tr 19 -vm 0 -vt 0  -ptb 0.5 -ca -ts 0
# python attack.py "topo-big2/ours-GAT-cite-ptb5" random_attack dev-7_Citeseer_GAT -tr 19 -vm 0 -vt 20 -ptb 0.5 -ca -ts 0
# python attack.py "topo-big2/base-GAT-cite-ptb10" random_attack dev-7_Citeseer_GAT -tr 19 -vm 0 -vt 0  -ptb 1.0 -ca -ts 0
# python attack.py "topo-big2/ours-GAT-cite-ptb10" random_attack dev-7_Citeseer_GAT -tr 19 -vm 0 -vt 20 -ptb 1.0 -ca -ts 0
# python attack.py "topo-big2/base-GAT-cite-ptb15"  random_attack dev-7_Citeseer_GAT -tr 19 -vm 0 -vt 0  -ptb 1.5 -ca -ts 0
# python attack.py "topo-big2/ours-GAT-cite-ptb15"  random_attack dev-7_Citeseer_GAT -tr 19 -vm 0 -vt 20 -ptb 1.5 -ca -ts 0
# python attack.py "topo-big2/base-GAT-cite-ptb20"  random_attack dev-7_Citeseer_GAT -tr 19 -vm 0 -vt 0  -ptb 2.0 -ca -ts 0
# python attack.py "topo-big2/ours-GAT-cite-ptb20"  random_attack dev-7_Citeseer_GAT -tr 19 -vm 0 -vt 20 -ptb 2.0 -ca -ts 0

# robust model experiment Cora
# python attack.py "model_big/base-ptb5"  random_attack dev-8_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.5 -ca -ts 0
# python attack.py "model_big/ours-ptb5"  random_attack dev-8_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.5 -ca -ts 5
# python attack.py "model_big/base-ptb10" random_attack dev-8_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 1.0  -ca -ts 0
# python attack.py "model_big/ours-ptb10" random_attack dev-8_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 1.0  -ca -ts 5
# python attack.py "model_big/base-ptb15" random_attack dev-8_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 1.5 -ca -ts 0
# python attack.py "model_big/ours-ptb15" random_attack dev-8_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 1.5 -ca -ts 5
# python attack.py "model_big/base-ptb20" random_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 2.0 -ca -ts 0
# python attack.py "model_big/ours-ptb20" random_attack dev-8_Cora_GCN -tr 19 -vm 20 -vt 0 -ptb 2.0 -ca -ts 5

# robust model experiment Citeseer
# python attack.py "model_big/base-citeseer-ptb5"  random_attack dev-7_Citeseer_GCN -vr 19 -vm  0 -vt 0 -ptb 0.5 -ca -ts 0
# python attack.py "model_big/ours-citeseer-ptb5"  random_attack dev-7_Citeseer_GCN -vr 19 -vm 20 -vt 0 -ptb 0.5 -ca -ts 5
# python attack.py "model_big/base-citeseer-ptb10" random_attack dev-7_Citeseer_GCN -vr 19 -vm  0 -vt 0 -ptb 1.0  -ca -ts 0
# python attack.py "model_big/ours-citeseer-ptb10" random_attack dev-7_Citeseer_GCN -vr 19 -vm 20 -vt 0 -ptb 1.0  -ca -ts 5
# python attack.py "model_big/base-citeseer-ptb15" random_attack dev-7_Citeseer_GCN -vr 19 -vm  0 -vt 0 -ptb 1.5 -ca -ts 0
# python attack.py "model_big/ours-citeseer-ptb15" random_attack dev-7_Citeseer_GCN -vr 19 -vm 20 -vt 0 -ptb 1.5 -ca -ts 5
# python attack.py "model_big/base-citeseer-ptb20" random_attack dev-7_Citeseer_GCN -tr 19 -vm  0 -vt 0 -ptb 2.0 -ca -ts 0
# python attack.py "model_big/ours-citeseer-ptb20" random_attack dev-7_Citeseer_GCN -tr 19 -vm 20 -vt 0 -ptb 2.0 -ca -ts 5