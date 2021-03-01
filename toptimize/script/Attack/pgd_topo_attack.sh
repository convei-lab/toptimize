# Adversarial attack with different edge perturbation ratios

# # robust topology experiment
# python attack.py "topo-t10/base-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.05 -ca -ts 0
# python attack.py "topo-t10/ours-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.05 -ca -ts 0
# python attack.py "topo-t10/base-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.1  -ca -ts 0
# python attack.py "topo-t10/ours-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.1  -ca -ts 0
# python attack.py "topo-t10/base-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.15 -ca -ts 0
# python attack.py "topo-t10/ours-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.15 -ca -ts 0
# python attack.py "topo-t10/base-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 0  -ptb 0.20 -ca -ts 0
# python attack.py "topo-t10/ours-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm 0 -vt 20 -ptb 0.20 -ca -ts 0

# robust model experiment
python attack.py "model-t10/base-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 0      
python attack.py "model-t10/ours-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.05 -ca -ts 5 -t 10
python attack.py "model-t10/base-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 0      
python attack.py "model-t10/ours-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.1  -ca -ts 5 -t 10
python attack.py "model-t10/base-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 0      
python attack.py "model-t10/ours-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.15 -ca -ts 5 -t 10
python attack.py "model-t10/base-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 0      
python attack.py "model-t10/ours-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.20 -ca -ts 5 -t 10

# robust model experiment
python attack.py "model-t8/base-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 0      
python attack.py "model-t8/ours-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.05 -ca -ts 5 -t 8
python attack.py "model-t8/base-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 0      
python attack.py "model-t8/ours-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.1  -ca -ts 5 -t 8
python attack.py "model-t8/base-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 0      
python attack.py "model-t8/ours-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.15 -ca -ts 5 -t 8
python attack.py "model-t8/base-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 0      
python attack.py "model-t8/ours-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.20 -ca -ts 5 -t 8

# robust model experiment
python attack.py "model-t12/base-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.05 -ca -ts 0      
python attack.py "model-t12/ours-ptb5"  pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.05 -ca -ts 5 -t 12
python attack.py "model-t12/base-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.1  -ca -ts 0      
python attack.py "model-t12/ours-ptb10" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.1  -ca -ts 5 -t 12
python attack.py "model-t12/base-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.15 -ca -ts 0      
python attack.py "model-t12/ours-ptb15" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.15 -ca -ts 5 -t 12
python attack.py "model-t12/base-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm  0 -vt 0 -ptb 0.20 -ca -ts 0      
python attack.py "model-t12/ours-ptb20" pgd_attack dev-7_Cora_GCN -vr 19 -vm 20 -vt 0 -ptb 0.20 -ca -ts 5 -t 12
