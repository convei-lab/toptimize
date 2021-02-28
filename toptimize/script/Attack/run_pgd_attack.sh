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

# #obust topology experiment
python attack.py base-ptb5 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0 -ptb 0.05
python attack.py ours-ptb5 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.05
python attack.py base-ptb10 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0 -ptb 0.1
python attack.py ours-ptb10 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.1
python attack.py base-ptb15 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0 -ptb 0.15
python attack.py ours-ptb15 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.15
python attack.py base-ptb20 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0 -ptb 0.20
python attack.py ours-ptb20 pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.20