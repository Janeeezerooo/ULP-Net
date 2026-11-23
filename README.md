# ULP-Net

Size-Agnostic Convolutional Neural Architecture for Linear Programming (LP) Unboundedness Detection.

## Structure
- `models/ulp_net.py`: ULP-Net architecture (`ColAvgPool` + Row/Col Global Conv)
- `data/generator.py`: LP $L_\infty$ normalization & canonical $L_2$ sorting
- `experiments/`: 10M-instance ablation scripts (Pooling & Activation)
