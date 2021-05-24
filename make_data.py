import pickle

with open('graph_features.pkl', 'rb') as f:
    graph_features = pickle.load(f)

print(graph_features)

for i, v in enumerate(graph_features):
    if i == 0:
        temp = v
    if i == 1:
        new = th.cat((temp,v), dim=0)
    if i > 1:
        new = th.cat((new, v), dim=0)
print(new, new.shape)