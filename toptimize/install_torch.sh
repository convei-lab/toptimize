export CUDA=cu101
# conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-geometric
