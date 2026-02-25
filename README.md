# Reproducing "Vision-Language Models are Zero-Shot Reward Models for RL"
## Course Project Guide

### Overview
This guide walks through reproducing key results from Rocamonde et al. (ICLR 2024).
We build a simplified, single-GPU implementation rather than using the authors' multi-GPU
Docker/Kubernetes infrastructure.

---

## Project Structure

```
vlm_rm_project/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── configs/
│   ├── cartpole.yaml
│   ├── mountaincar.yaml
│   └── humanoid_kneel.yaml
├── src/
│   ├── vlm_reward.py           # CLIP reward model + goal-baseline regularization
│   ├── environments.py          # Environment wrappers (textures, camera mods)
│   ├── train_classic.py         # CartPole / MountainCar experiments
│   ├── train_humanoid.py        # Humanoid experiments
│   ├── evaluate.py              # EPIC distance + reward landscape plotting
│   └── utils.py                 # Rendering, logging helpers
└── results/
    └── plots/
```

---

## Phase 0: Environment Setup

### Option A: Local machine with GPU (recommended for CartPole/MountainCar)
### Option B: GCP VM with GPU (needed for humanoid with ViT-bigG-14)

For GCP, a single **NVIDIA T4** (~$0.35/hr) works for smaller CLIP models.
For ViT-bigG-14 you'll want an **A100 40GB** (~$3.67/hr) or **L4** (~$0.70/hr).

```bash
# Create a conda environment
conda create -n vlmrm python=3.10 -y
conda activate vlmrm

# Core dependencies
pip install torch torchvision  # match your CUDA version
pip install open_clip_torch    # for CLIP models
pip install gymnasium[mujoco]  # MuJoCo environments
pip install stable-baselines3  # RL algorithms
pip install mujoco             # MuJoCo physics engine
pip install matplotlib numpy scipy pandas
pip install wandb              # optional, for logging
pip install Pillow imageio     # for rendering/video

# Verify MuJoCo works
python -c "import gymnasium; env = gymnasium.make('Humanoid-v4'); print('MuJoCo OK')"

# Verify CLIP works
python -c "import open_clip; model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k'); print('CLIP OK')"
```

---

## Phase 1: CartPole Reward Landscape

**Goal:** Verify CLIP cosine similarity correlates with ground truth reward.
No RL training needed — just render frames and compute CLIP rewards.

### What to implement:
1. Render CartPole at different pole angles
2. Compute CLIP similarity with "pole vertically upright on top of the cart"
3. Plot reward vs. pole angle (reproduce Figure 2a)
4. Repeat with goal-baseline regularization at different α values

### Expected result:
- Peak reward at pole angle ≈ 0 (upright)
- Smooth, well-shaped reward landscape

---

## Phase 2: MountainCar with Textures

**Goal:** Show that more realistic rendering improves CLIP reward quality.

### What to implement:
1. Default MountainCar: render at different x positions, compute CLIP reward
2. Textured MountainCar: add mountain/sky texture, repeat
3. Compare reward landscapes (reproduce Figures 2b vs 2c)
4. Train DQN/SAC agents with CLIP reward on both versions

### Key insight to verify:
- Default rendering → noisy, poorly shaped rewards
- Textured rendering + regularization → well-shaped, learnable rewards

---

## Phase 3: Humanoid Tasks (Days 3-7)

**Goal:** Train humanoid to kneel, do splits, sit in lotus position using CLIP rewards.

### Environment modifications (critical!):
1. **Textures:** Change humanoid skin color and background to be more realistic
2. **Camera:** Fixed position, slightly angled down (not following the agent)

### Tasks to attempt (in order of expected ease):
1. **Kneeling** — "a humanoid robot kneeling" (100% success in paper)
2. **Doing splits** — "a humanoid robot practicing gymnastics, doing the side splits" (100%)
3. **Lotus position** — "a humanoid robot seated down, meditating in the lotus position" (100%)

### Training details (from paper Appendix C.2):
- Algorithm: SAC
- Steps: 10M (can try fewer first, e.g., 2-5M, to see if learning signal emerges)
- Episode length: 100
- Learning starts: 50,000 steps
- SAC updates: 100 every 100 env steps
- τ = 0.005, γ = 0.95, lr = 6e-4
- CLIP model: ViT-bigG-14 (critical — smaller models get 0% success)
- Batch size for CLIP inference: 3200

### Compute estimate:
- With single A100: ~12-24 hours per task per seed
- With T4: likely too slow for ViT-bigG-14 (model is ~2.5B params)
- Start with 1 seed, expand to 2-4 if time permits

---

## Phase 4: Ablations (Days 7-9)

### 4a: Goal-Baseline Regularization
- Run kneeling task with α ∈ {0, 0.25, 0.5, 0.75, 1.0}
- Compute EPIC distance to human labels
- Reproduce Figure 4a

### 4b: Model Scale (if compute allows)
- Run kneeling with RN50, ViT-L-14, ViT-H-14, ViT-bigG-14
- Compare EPIC distances and success rates
- Reproduce Figure 4b,c

---

## Phase 5: Extension (Days 9-12)

### Suggested extensions (pick one):

**A. Newer CLIP Models (easiest, most interesting)**
- Try SigLIP, MetaCLIP, or EVA-CLIP as reward models
- These didn't exist when the paper was written
- Directly tests their "scaling hypothesis" — do better VLMs = better rewards?
- Compare EPIC distances and success rates against ViT-bigG-14

**B. Prompt Sensitivity Analysis**
- For kneeling task, try 10+ prompt variations:
  - "a humanoid robot kneeling"
  - "a robot on its knees"
  - "kneeling position"
  - "a person kneeling on the ground"
  - etc.
- Measure how reward quality varies with prompt choice
- Authors claim "no prompt engineering" but never ablate this

**C. New Task**
- Try a task not in the paper: "a humanoid robot crawling",
  "a humanoid robot lying down", "a humanoid robot squatting"
- Report success/failure and analyze why

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
