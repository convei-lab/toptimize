# Adversarial attack with different edge perturbation ratios

# # robust topology experiment 1
# python attack.py "topo-t10/base-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.05 -ca -ts 0
# python attack.py "topo-t10/ours-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.05 -ca -ts 0
# python attack.py "topo-t10/base-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.1  -ca -ts 0
# python attack.py "topo-t10/ours-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.1  -ca -ts 0
# python attack.py "topo-t10/base-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.15 -ca -ts 0
# python attack.py "topo-t10/ours-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.15 -ca -ts 0
# python attack.py "topo-t10/base-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.20 -ca -ts 0
# python attack.py "topo-t10/ours-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.20 -ca -ts 0

# robust topology experiment Citeseer
# python attack.py "topo-citeseer/base-ptb5"  pgd_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 0  -ptb 0.05 -ca -ts 0
# python attack.py "topo-citeseer/ours-ptb5"  pgd_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 20 -ptb 0.05 -ca -ts 0
# python attack.py "topo-citeseer/base-ptb10" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 0  -ptb 0.1  -ca -ts 0
# python attack.py "topo-citeseer/ours-ptb10" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 20 -ptb 0.1  -ca -ts 0
# python attack.py "topo-citeseer/base-ptb15" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 0  -ptb 0.15 -ca -ts 0
# python attack.py "topo-citeseer/ours-ptb15" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 20 -ptb 0.15 -ca -ts 0
# python attack.py "topo-citeseer/base-ptb20" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 0  -ptb 0.20 -ca -ts 0
# python attack.py "topo-citeseer/ours-ptb20" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm 0 -vt 20 -ptb 0.20 -ca -ts 0

# robust model experiment 2
# python attack.py "model-t10/base-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 0      
# python attack.py "model-t10/ours-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.05 -ca -ts 5 -t 10
# python attack.py "model-t10/base-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 0      
# python attack.py "model-t10/ours-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.1  -ca -ts 5 -t 10
# python attack.py "model-dev7_11/base-ptb15" pgd_attack dev-7_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 0       
# python attack.py "model-dev7_11/ours-ptb15" pgd_attack dev-7_Cora_GCN -tr 19 -vm 0 -vt 0 -ptb 0.15 -ca -ts 5 -t 11
# python attack.py "model-dev8/base-ptb20" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 0      
# python attack.py "model-dev8/ours-ptb20" pgd_attack dev-8_Cora_GCN -tr 19 -vm 20 -vt 0 -ptb 0.20 -ca -ts 5 -t 10

# robust model experiment Citeseer
# python attack.py "model-citeseer/base-ptb5"  pgd_attack dev-7_Citeseer_GCN -tr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 0      
# python attack.py "model-citeseer/ours-ptb5"  pgd_attack dev-7_Citeseer_GCN -tr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 5 -t 7
# python attack.py "model-citeseer/base-ptb10" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 0      
# python attack.py "model-citeseer/ours-ptb10" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 5 -t 7
# python attack.py "model-citeseer/base-ptb15" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 0      
# python attack.py "model-citeseer/ours-ptb15" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 5 -t 7
# python attack.py "model-citeseer/base-ptb20" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 0      
# python attack.py "model-citeseer/ours-ptb20" pgd_attack dev-7_Citeseer_GCN -tr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 5 -t 7

# robust model experiment
# python attack.py "model-t8/base-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 0      
# python attack.py "model-t8/ours-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.05 -ca -ts 5 -t 8
# python attack.py "model-t8/base-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 0      
# python attack.py "model-t8/ours-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.1  -ca -ts 5 -t 8
# python attack.py "model-t8/base-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 0      
# python attack.py "model-t8/ours-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.15 -ca -ts 5 -t 8
# python attack.py "model-t8/base-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 0      
# python attack.py "model-t8/ours-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.20 -ca -ts 5 -t 8

# robust model experiment
# python attack.py "model-t12/base-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 0      
# python attack.py "model-t12/ours-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.05 -ca -ts 5 -t 12
# python attack.py "model-t12/base-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 0      
# python attack.py "model-t12/ours-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.1  -ca -ts 5 -t 12
# python attack.py "model-t12/base-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 0      
# python attack.py "model-t12/ours-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.15 -ca -ts 5 -t 12
# python attack.py "model-t12/base-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 0      
# python attack.py "model-t12/ours-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.20 -ca -ts 5 -t 12

# robust model experiment
# python attack.py "model-8/base-ptb5"  pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 0      
# python attack.py "model-8/ours-ptb5"  pgd_attack dev-8_Cora_GCN -tr 19 -vm 20 -vt 0 -ptb 0.05 -ca -ts 5 
# python attack.py "model-8/base-ptb10" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 0      
# python attack.py "model-8/ours-ptb10" pgd_attack dev-8_Cora_GCN -tr 19 -vm 20 -vt 0 -ptb 0.1  -ca -ts 5
# python attack.py "model-8/base-ptb15" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 0      
# python attack.py "model-8/ours-ptb15" pgd_attack dev-8_Cora_GCN -tr 19 -vm 20 -vt 0 -ptb 0.15 -ca -ts 5
# python attack.py "model-8/base-ptb20" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 0
# python attack.py "model-8/ours-ptb20" pgd_attack dev-8_Cora_GCN -tr 19 -vm 20 -vt 0 -ptb 0.20 -ca -ts 5 -ea # t, ea, l1, hs, l2, lr, ts
# python attack.py "model-test/base-ptb20" pgd_attack dev-8_Cora_GCN -tr 1 -vm  0 -vt 0 -ptb 0.20 -ca -ts 3 -et     
# python attack.py "model-test/ours-ptb20" pgd_attack dev-8_Cora_GCN -tr 1 -vm 20 -vt 0 -ptb 0.20 -ca -ts 3 -et

# Compare test node masked topology
# python attack.py "model-test/base-ptb20-test" pgd_attack dev-8_Cora_GCN -tr 1 -vm  0 -vt 0 -ptb 0.20 -ca -ts 1 -et     
# python attack.py "model-test/ours-ptb20-test" pgd_attack dev-8_Cora_GCN -tr 1 -vm 20 -vt 0 -ptb 0.20 -ca -ts 1 -et

# Compare train node masked topology
# python attack.py "model-test/base-ptb20-wnb" pgd_attack dev-8_Cora_GCN -tr 1 -vm  0 -vt 0 -ptb 0.20 -ca -ts 1 -et -wnb
# python attack.py "model-test/ours-ptb20-wnb" pgd_attack dev-8_Cora_GCN -tr 1 -vm 20 -vt 0 -ptb 0.20 -ca -ts 1 -et -wnb

# python attack_gat.py "PA1-cora-GAT-ptb5"  pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 0      
# python attack_gat.py "PA1-cora-GAT-ptb5"  pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 5 -t 10
# python attack_gat.py "PA1-cora-GAT-ptb10" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 0      
# python attack_gat.py "PA1-cora-GAT-ptb10" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 5 -t 10
python attack_gat.py "PA1-cora-base-ptb15" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 0       
python attack_gat.py "PA1-cora-ours-ptb15" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 5 -t 10
python attack_gat.py "PA1-cora-base-ptb20" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 0      
python attack_gat.py "PA1-cora-ours-ptb20" pgd_attack dev-8_Cora_GCN -tr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 5 -t 10