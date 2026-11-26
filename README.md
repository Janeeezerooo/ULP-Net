# ULP-Net: A Deep Learning Approach for Detecting Unbounded Linear Programming Problems

Official implementation of **ULP-Net**, a resolution-independent deep learning architecture for identifying feasible, unbounded Linear Programming (LP) problems and pinpointing the decision variables forming unbounded recession rays.

## Repository Structure
- `models/ulp_net.py`: ULP-Net architecture (151,041 parameters) featuring infinity-norm scaling, canonical L2-norm sorting, $1\times 1$ axial mean pooling, two $3\times 3$ `LeakyReLU(0.01)` ConvBlocks, and column-wise adaptive pooling (`ColPool`).
- `data/generator.py`: 4-regime synthetic LP generator (Dense, High-Sparsity, Extreme-Scale $10^{-3}$--$10^4$, and Oblique Multi-Variable Recession Rays with hard-negative decoy columns).
- `data/netlib/`: Standard real-world Netlib LP benchmark instances (`AFIRO`, `SC50B`, `ADLITTLE`, `BLEND`, `SHARE2B`).
- `experiments/`: Component-level 10M ablation scripts (`ablation_pooling.py`, `ablation_activation.py`, `ablation_architecture_real.py`, `ablation_depth_10M.py`), multi-scale evaluation benchmarks (`final_paper_suite.py`), and large-scale training scripts (`large_scale/`).
- `checkpoints/`: Pre-trained model weights (`.pt`).

## Quick Start
```bash
pip install -r requirements.txt
python experiments/final_paper_suite.py
```
