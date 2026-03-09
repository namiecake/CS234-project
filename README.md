# Reproducing "Vision-Language Models are Zero-Shot Reward Models for RL"
## CS 234 Course Project

### Overview
This project reproduces key results from Rocamonde et al. (ICLR 2024), which
demonstrates that CLIP cosine similarity can serve as a zero-shot reward signal
for reinforcement learning — no hand-designed reward functions needed. We build
a simplified, single-GPU implementation rather than using the authors' multi-GPU
Docker/Kubernetes infrastructure.

As an extension, we also evaluate **SigLIP2** (ViT-SO400M-14-SigLIP2) as a
drop-in replacement for the paper's CLIP ViT-bigG-14 reward model, testing
whether newer and smaller vision-language models improve reward quality.

## Project Structure

```
CS234-project/
├── README.md                       # This file
├── requirements.txt                # Dependencies
├── data/
│   └── humanoid_textured.xml       # MuJoCo humanoid model (sky, floor, body textures)
├── src/
│   ├── vlm_reward.py               # CLIP/SigLIP reward model + goal-baseline regularization
│   ├── environments.py             # Environment wrappers (textures, camera mods)
│   ├── train_classic.py            # CartPole / MountainCar reward landscape experiments
│   ├── train_humanoid.py           # Humanoid SAC training with VLM rewards
│   ├── evaluate.py                 # EPIC distance + model-scale comparison
│   └── test_humanoid_setup.py      # Quick sanity check for MuJoCo + CLIP pipeline
└── results/                        # Training outputs (checkpoints, TensorBoard, videos)
```

---

## Setup

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU for humanoid experiments (we use nvidia L4)
- CPU is sufficient for CartPole/MountainCar reward-landscape experiments with smaller models (RN50)

### Installation

```bash
conda create -n vlmrm python=3.10 -y
conda activate vlmrm

pip install -r requirements.txt
# If needed, install a CUDA-matched PyTorch first:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify the setup

```bash
# MuJoCo
python -c "import gymnasium; env = gymnasium.make('Humanoid-v4'); print('MuJoCo OK')"

# CLIP (downloads ~10 GB for ViT-bigG-14 on first run)
python -c "import open_clip; model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k'); print('CLIP OK')"

# End-to-end pipeline check
python src/test_humanoid_setup.py
```

---

## Experiments

### 1. CartPole Reward Landscape

Verify that CLIP cosine similarity correlates with ground-truth reward (reproduces Figure 2a).
No RL training — just render CartPole at different pole angles and plot CLIP reward.

```bash
python src/train_classic.py --experiment cartpole --model RN50
```

**Expected result:** peak reward at pole angle ≈ 0 (upright), smooth landscape.

### 2. MountainCar with Textures

Show that more realistic rendering improves CLIP reward quality (reproduces Figures 2b/c).

```bash
python src/train_classic.py --experiment mountaincar --model RN50
```

**Key finding:** default rendering produces noisy rewards; textured rendering + regularization yields a well-shaped, learnable reward landscape.

### 3. Humanoid Tasks (main experiment)

Train a MuJoCo humanoid to perform tasks specified only by text prompts, using
VLM cosine similarity as the sole reward signal.

#### Environment setup

The humanoid uses a textured MuJoCo XML model (`data/humanoid_textured.xml`)
with a fixed camera (distance 3.5, elevation −10°, azimuth 180°) — both
modifications are critical for CLIP to produce usable rewards (Section 4.3).

#### Tasks we ran

We ran **kneeling** and **splits** with two VLM backbones:

| Task | VLM Backbone | Pretrained Weights | Paper Success |
|---|---|---|---|
| Kneeling | **ViT-bigG-14** | `laion2b_s39b_b160k` | 100% |
| Splits | **ViT-bigG-14** | `laion2b_s39b_b160k` | 100% |
| Kneeling | **SigLIP2-SO400M** | `webli` | — (extension) |
| Splits | **SigLIP2-SO400M** | `webli` | — (extension) |

```bash
# Kneeling with ViT-bigG-14 (paper baseline)
python src/train_humanoid.py --task kneeling --model ViT-bigG-14 --total_steps 10000000

# Splits with ViT-bigG-14
python src/train_humanoid.py --task splits --model ViT-bigG-14 --total_steps 10000000

# Kneeling with SigLIP2 (extension)
python src/train_humanoid.py --task kneeling --model SigLIP2-SO400M --total_steps 10000000

# Splits with SigLIP2 (extension)
python src/train_humanoid.py --task splits --model SigLIP2-SO400M --total_steps 10000000
```

#### Training details (from paper, Appendix C.2)

- Algorithm: SAC
- Steps: 10M
- Episode length: 100
- Learning starts: 50,000 steps
- SAC updates: 100 every 100 env steps (train_freq=100, gradient_steps=100)
- τ = 0.005, γ = 0.95, lr = 6e-4
- CLIP model: ViT-bigG-14 (~2.5B params, ~10 GB VRAM)

Rewards are computed in batched mode (Algorithm 1 from the paper): the
environment buffers rendered frames during each episode, and a callback runs
a single CLIP forward pass at the end of each rollout before patching the
SAC replay buffer.

#### Compute estimate

- Single A100: ~12–24 hours per task per seed
- T4: likely too slow for ViT-bigG-14

#### Evaluating a trained model

```bash
python src/train_humanoid.py --eval_only results/<run_dir>/checkpoints/final_model --task kneeling
```

This renders 20 episodes, computes mean CLIP reward, and saves videos.

### 4. Ablations

#### Goal-Baseline Regularization

The codebase supports goal-baseline regularization (Definition 1 from the paper)
controlled by the `--alpha` flag. To sweep:

```bash
for alpha in 0.0 0.25 0.5 0.75 1.0; do
  python src/train_humanoid.py --task kneeling --alpha $alpha
done
```

#### Model Scale Comparison

Compare EPIC distances across model sizes (reproduces Figure 4):

```bash
python src/evaluate.py --experiment scale
```

This evaluates RN50, ViT-L-14, ViT-H-14, and ViT-bigG-14 reward quality on
collected frames.

### 5. Extension: SigLIP2 as a Reward Model

The paper's central claim is that VLM scale drives reward quality — bigger
models yield better-shaped rewards. We test this by swapping in
**SigLIP2-SO400M** (ViT-SO400M-14-SigLIP2, pretrained on WebLI), a newer
vision-language model that did not exist when the paper was written. This
directly probes whether improvements in VLM architecture and training data
translate to better zero-shot reward signals.

We ran SigLIP2 on the **kneeling** and **splits** tasks using the same
hyperparameters as the ViT-bigG-14 experiments.

---

## Supported VLM Backbones

All models are loaded via [OpenCLIP](https://github.com/mlfoundations/open_clip):

| Key | Model | Pretrained | Params | Notes |
|---|---|---|---|---|
| `RN50` | ResNet-50 | `openai` | 102M | Fast CPU testing |
| `ViT-L-14` | ViT-L/14 | `laion2b_s32b_b82k` | 428M | Mid-size |
| `ViT-H-14` | ViT-H/14 | `laion2b_s32b_b79k` | 986M | Large |
| `ViT-bigG-14` | ViT-bigG/14 | `laion2b_s39b_b160k` | 2.5B | Paper's primary model |
| `SigLIP2-SO400M` | ViT-SO400M-14-SigLIP2 | `webli` | 400M | Extension model |

---

## Available Humanoid Tasks

From the paper (Table 3). All tasks use baseline prompt *"a humanoid robot"*:

| Task | Goal Prompt | Paper Success Rate |
|---|---|---|
| `kneeling` | "a humanoid robot kneeling" | 100% |
| `splits` | "a humanoid robot practicing gymnastics, doing the side splits" | 100% |
| `lotus` | "a humanoid robot seated down, meditating in the lotus position" | 100% |
| `standing` | "a humanoid robot standing up" | 100% |
| `arms_raised` | "a humanoid robot standing up, with both arms raised" | 100% |
| `hands_on_hips` | "a humanoid robot standing up with hands on hips" | 64% |
| `arms_crossed` | "a humanoid robot standing up, with its arms crossed" | 0% |
| `one_leg` | "a humanoid robot standing up on one leg" | 0% |

---

## Key Implementation Notes

### CLIP Reward Computation (Equation 2 from paper)
```python
R_CLIP(s) = cos_sim(CLIP_L(text), CLIP_I(image))
```

### Goal-Baseline Regularization (Definition 1)
```python
# g = normalized text encoding of goal prompt
# b = normalized text encoding of baseline prompt
# s = normalized image encoding of state
# L = line spanned by b and g
# α controls regularization strength
R_reg(s) = 1 - 0.5 * ||α * proj_L(s) + (1-α) * s - g||²
```

### EPIC Distance (for evaluation)
```python
# For goal-based tasks, EPIC simplifies to Pearson correlation
# between CLIP rewards and binary human labels
D_EPIC = (1/√2) * √(1 - ρ(R_CLIP, R_human))
```
