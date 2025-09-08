# Stepwise Generative–Discriminative Reasoning Framework for Metasurface Inverse Design

## Introduction and Information

This repository hosts the reference implementation of the algorithms described in the manuscript **“Stepwise Generative–Discriminative Reasoning Framework for Metasurface Inverse Design”**. It documents the dataset, model architectures, the design rule check (DRC) algorithm, and the evaluation protocol to support scientific reproducibility and facilitate comprehension. For intellectual-property and security considerations, code related to data visualization is not included in this release.
- Title: Stepwise generative-discriminative reasoning framework for metasurface inverse design
- Authors: Jiakang Lao, Zihan Zhao, Shuyang Zhou, Guoqin Cao, Zhihang Jiang, Cong Wang, Xinwei Wang, Xumin Ding
- Published to: Under review
- DOI: Under review

## Project Structure

<pre>
.
├── dataset/                              # Dataset
│   ├── dataset.py                        # Dataloader
│   └── freeform.mat                      # Dataset (MATLAB .mat)
│
├── diffusion/                            # Diffusion implementations
│   ├── diffusion.py                      # Diffusion (without parameters)
│   └── diffusion_baseline.py             # Diffusion (with parameters)
│
├── models/                               # Model architectures
│   ├── pattern_unet.py                   # Pattern U-Net (without parameters)
│   ├── pattern_unet_baseline.py          # Pattern U-Net (with parameters)
│   ├── parameter_transformer.py          # Parameter Transformer
│   ├── parameter_pnn.py                  # Parameter PNN
│   └── transmittance_transformer.py      # Transmittance Transformer
│
├── infer/                                # Inference scripts
│   ├── infer_pattern.py                  # Pattern generation (without parameters)
│   ├── infer_pattern_baseline.py         # Pattern generation (with parameters)
│   ├── infer_parameter.py                # Parameter inference
│   ├── infer_transmittance.py            # Transmittance prediction
│   └── infer_all.py                      # End-to-end MetaSR pipeline
│
├── train_pattern_unet.py                 # Train Pattern U-Net (without parameters)
├── train_pattern_unet_baseline.py        # Train Pattern U-Net (with parameters)
├── train_parameter_transformer.py        # Train Parameter Transformer
├── train_parameter_pnn.py                # Train Parameter PNN
├── train_transmittance_transformer.py    # Train Transmittance Transformer
├── requirements.txt                      # dependencies
└── README.md                             # documentation
</pre>


## Appendix S0. Instructions

- **Environment Setup**
  - Python ≥ 3.12.7
  - CUDA ≥ 12.4 with a compatible GPU
  - install the dependencies by running `pip install -r requirements.txt`
- **Training Procedures**
  - Run all scripts from the repository root to train the models:
    - Pattern U-Net: `train_pattern_unet.py` or `train_pattern_unet_baseline.py`
    - Parameter Transformer: `train_parameter_transformer.py`
    - Parameter PNN: `train_parameter_pnn.py`
    - Transmittance Transformer: `train_transmittance_transformer.py`
  - Hyperparameters are defined in each script and can be edited directly.
  - Checkpoints and loss data are saved to `results/` by default.

## Appendix S1. Dataset Preparation and Analysis

MetaSR utilizes an open-source dataset of 174,883 samples to enable direct comparison with MetaDiffusion and I-P DM baselines. The dataset requires no additional preprocessing and can be used immediately upon download.

- **Acquisition and setup**
  - Download `freeform.mat` from the [Meta-atoms Data Sharing Repository](https://github.com/SensongAn/Meta-atoms-data-sharing)
  - Place the file in the in the `dataset/` directory under project root
  - The default dataloader `dataset/dataset.py` automatically detects this path
- **Dataset specification**
  - Structural patterns: Binary patterns describing high-index dielectric occupancy stored as 1×64×64 matrices. Due to symmetry constraints, the effective design space reduces to 1×32×32 patterns.
  - Structural parameters: Three continuous design variables define the parameter space:
     - Lattice size (L): 2.5-3.0 μm
     - Thickness (T): 0.5-1.0 μm
     - Refractive index (R): 3.5-5.0
- **Implementation note**
  The benchmark dataset models refractive index as continuous for consistency with prior literature. For optimal deployment, MetaSR should train the Parameter Transformer on discrete values from manufacturable material libraries to eliminate discretization losses and facilitate fabrication integration.
- **Statistical analysis and diversity**
  Pattern diversity is quantified through fill ratio distributions shown in Fig. S1. After removing duplicates, 168,227 unique patterns exhibit fill ratios spanning 0–0.65, demonstrating comprehensive coverage of the design space with sufficient variability for robust model training.

<div align="center">
    <img src="README/Fig S1.png" width="70%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S1.</strong> Distribution of pattern fill ratios of dataset.
</p>

## Appendix S2. Convergence Proof of Diffusion Models

We provide a concise SDE-based argument for DDPM convergence, matching the main-text notation and connecting the practical denoising loss to a theoretical KL bound.

- **Overview**
  - The discrete forward process is the Euler–Maruyama discretization of a variance-preserving SDE with schedule $\beta(t)$, which drives any initial distribution to an isotropic Gaussian.
  - The reverse process is the time-reversed SDE. When the score $\nabla_x \log p_t(x)$ is accurately approximated, sampling the reverse SDE recovers the data distribution.
  - A KL bound via Girsanov's theorem links generation error to the integrated score-matching error.
- **Forward SDE**  
  Let $w(t)$ be standard Brownian motion. The forward SDE is
  $$\mathrm{d}x = -\frac{1}{2}\beta(t)x\mathrm{d}t + \sqrt{\beta(t)}\mathrm{d}w(t). \tag{S1}$$
  The marginal $p_t(x)$ satisfies the Fokker–Planck equation
  $$\frac{\partial p_t}{\partial t} = \frac{\beta(t)}{2}\nabla \cdot (x p_t) + \frac{\beta(t)}{2}\Delta p_t. \tag{S2}$$
  Consequently,
  $$\lim_{t\to\infty} p_t(x) = \mathcal{N}(0, I), \tag{S3}$$
  establishing convergence of the forward process.
- **Reverse SDE and score connection**  
  By Anderson's time-reversal,
  $$\mathrm{d}x = \left[-\frac{1}{2}\beta(t)x - \beta(t)\nabla_x \log p_t(x)\right]\mathrm{d}t + \sqrt{\beta(t)}\mathrm{d}\bar{w}(t). \tag{S4}$$
  In the DDPM parameterization, the denoiser $\epsilon_\theta(x_t,t)$ induces a score estimate
  $$\nabla_x \log p_t(x) \approx s_\theta(x,t) \simeq -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}, \tag{S5}$$
  where $\bar{\alpha}_t=\prod_{s\leq t}(1-\beta_s)$ is the discrete cumulative product.
- **Convergence bound**  
  Comparing the exact reverse SDE with score $\nabla_x \log p_t(x)$ and the approximate one with $s_\theta(x,t)$, Girsanov's theorem yields
  $$D_{\mathrm{KL}}(p_0 \parallel p_\theta) \leq C \int_0^T \mathbb{E}_{p_t}\left[ \left\|\nabla_x \log p_t(x) - s_\theta(x,t)\right\|^2 \right]\mathrm{d}t, \tag{S6}$$
  where $C$ depends on the noise schedule. Hence, as the integrated score error vanishes, the KL divergence goes to zero and $p_\theta \to p_0$.
- **Implication for the DDPM loss**  
  The standard objective
  $$\mathcal{L}_{\mathrm{DDPM}} = \mathbb{E}_{t,x_0,\epsilon,c}\left[ \left\|\epsilon_t - \epsilon_\theta(x_t,t,c)\right\|^2 \right] \tag{S7}$$
  With sufficient capacity and convergence, often aided by conditioning $c$, $s_\theta(x,t)$ approaches the true score and sampling the reverse SDE recovers the data distribution with quantifiable error.

## Appendix S3. Pattern U-Net Variants: Architecture and Training Results

The Pattern Generation Module employs a specialized Pattern U-Net architecture optimized through variant comparison. All variants follow U-shaped encoder–decoder designs conditioned on transmittance and timestep, processing 1×32×32 pattern noise through multi-resolution stages. Architecture specifications are detailed in Fig. S2 and Table S1.

<div align="center">
    <img src="README/Fig S2.svg" width="70%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S2.</strong> Architecture of Pattern U-Net variants.
</p>

**Table S1.** Pattern U-Net variants: architecture specifications.
| Stage         | Operation           | Input dims       | Output dims      |
| :------------ | :------------------ | :--------------- | :--------------- |
| Input         | –                   | [B, 1, 32, 32]   | [B, 1, 32, 32]   |
| Encoder1      | Convolutional Block | [B, 1, 32, 32]   | [B, 64, 32, 32]  |
| DownSampling1 | MaxPool2d           | [B, 64, 32, 32]  | [B, 64, 16, 16]  |
| Encoder2      | Convolutional Block | [B, 64, 16, 16]  | [B, 128, 16, 16] |
| DownSampling2 | MaxPool2d           | [B, 128, 16, 16] | [B, 128, 8, 8]   |
| Encoder3      | Convolutional Block | [B, 128, 8, 8]   | [B, 256, 8, 8]   |
| DownSampling3 | MaxPool2d           | [B, 256, 8, 8]   | [B, 256, 4, 4]   |
| Encoder4      | Convolutional Block | [B, 256, 4, 4]   | [B, 512, 4, 4]   |
| DownSampling4 | MaxPool2d           | [B, 512, 4, 4]   | [B, 512, 2, 2]   |
| Middle        | Convolutional Block | [B, 512, 2, 2]   | [B, 1024, 2, 2]  |
| UpSampling4   | Upsample            | [B, 1024, 2, 2]  | [B, 1024, 4, 4]  |
| Decoder4      | Convolutional Block | [B, 1536, 4, 4]  | [B, 512, 4, 4]   |
| UpSampling3   | Upsample            | [B, 512, 4, 4]   | [B, 512, 8, 8]   |
| Decoder3      | Convolutional Block | [B, 768, 8, 8]   | [B, 256, 8, 8]   |
| UpSampling2   | Upsample            | [B, 256, 8, 8]   | [B, 256, 16, 16] |
| Decoder2      | Convolutional Block | [B, 384, 16, 16] | [B, 128, 16, 16] |
| UpSampling1   | Upsample            | [B, 128, 16, 16] | [B, 128, 32, 32] |
| Decoder1      | Convolutional Block | [B, 192, 32, 32] | [B, 64, 32, 32]  |
| Output        | Convolutional Layer | [B, 64, 32, 32]  | [B, 1, 32, 32]   |

Four variants are evaluated after 200 training epochs, with performance metrics summarized in Table S2. Results include model parameters, training and validation losses, and inference speed under batch size 256.

**Table S2.** Pattern U-Net variants: performance comparison
| Item                         | Model 1 | Model 2 | Model 3 | Model 4 |
|:-----------------------------|:--------|:--------|:--------|:--------|
| Parameter input              | Yes     | No      | Yes     | No      |
| Cross-attention              | No      | No      | Yes     | Yes     |
| Model parameters             | 34,920,385 | 34,911,553 | 49,295,329 | 49,268,417 |
| Training loss                | 0.005341 | 0.005779 | 0.004782 | 0.005194 |
| Validation loss              | 0.013420 | 0.017110 | 0.005041 | 0.005335 |
| Inference speed (s/sample)   | 0.4265  | 0.4262  | 0.1397  | 0.1373  |

Cross-attention integration delivers superior performance with minimal overfitting, as evidenced by closely matched training and validation losses. Despite a 41% parameter increase, cross-attention variants achieve 3.05× faster inference, demonstrating enhanced efficiency and effectiveness in the Pattern U-Net architecture.

## Appendix S4. Design Rule Check (DRC) Algorithm

MetaSR incorporates a DRC procedure to ensure pattern manufacturability through connected-component analysis using `scipy.ndimage.label` with 8-connected neighborhoods.

- **Fabrication Constraints**
  DRC parameters align with Nanoscribe Photonic Professional GT capabilities:
  - Lateral resolution: 400 nm minimum
  - Vertical resolution: 20 nm minimum
  - structure height: 50 μm maximum
  - feature spacing: 200 nm minimum
- **Parameter Mapping**
  For 64×64-pixel unit cells with lattice constants of 2.5–3.0 μm, the pixel-to-physical ratio spans 39–47 nm per pixel, ensuring feature constraints exceed fabrication limits with safety margins.
- **DRC Implementation**
  Two primary constraints filter invalid designs before parameter inference:
  - Maximum 8 connected components per unit cell
  - Minimum feature size of 10 pixels
- **Implementation Benefits**
  The DRC suppresses isolated-pixel artifacts, promotes manufacturability, and reduces computational overhead by pre-screening designs. This approach effectively bridges computational design with micro/nanofabrication constraints within the stepwise reasoning framework.

## Appendix S5. Parameter Transformer: Architecture and Training Results

The Parameter Inference Module employs a specialized Parameter Transformer based on Transformer-Encoder architecture to map pattern-transmittance pairs to structural parameters. See Fig. S3 for the architectural overview.

<div align="center">
    <img src="README/Fig S3.svg" width="70%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S3.</strong> Parameter Transformer’s architecture.
</p>

- **Architecture**
  - Embedding Layer: Processes 1×32×32 patterns and 2×301 transmittance spectra through separate linear projections to 1024 dimensions, concatenates to 2048 dimensions, then maps to unified 1024-dimensional tokens
  - Transformer Encoder: Processes 1024-dimensional tokens through 8-layer Transformer with 16-head attention, capturing long-range dependencies and contextual relationships within the pattern-transmittance data
  - Feedforward Network: Maps encoded features to 1×3 output vector containing inferred structural parameters L, T, and R

<div align="center">
    <img src="README/Fig S4.svg" width="60%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S4.</strong> Training and validation loss curves of the Parameter Transformer
</p>

- **Training Results**
  After 200 epochs, the model achieved robust parameter inference with minimal overfitting, as shown in Fig. S4.
- **Loss Metrics and Reproducibility**
  - Seed 42: Training loss 0.0007855, validation loss 0.003264
  - Seed 3407: Training loss 0.0007670, validation loss 0.002662
  - To ensure reproducibility and avoid selection bias, all main-text results use seed 42, while noting marginally superior performance with seed 3407. This demonstrates architectural robustness across random initializations

## Appendix S6. Parameter PNN: Architecture and Training Results

To benchmark the Parameter Transformer, we adapt the PNN as a baseline for parameter inference. See Fig. S5 for the architectural overview.

<div align="center">
    <img src="README/Fig S5.svg" width="70%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S5.</strong> Parameter PNN’s architecture.
</p>

- **Architecture**
  - Image Processing Network: Processes 1×32×32 patterns through three convolutional blocks with progressive downsampling to 64×4×4 features
  - Transmittance Processing Network: Transforms 2×301 transmittance spectra via MLP and transposed convolution to matching 64×4×4 features
  - Parameter Prediction Network: Processes fused 128×4×4 features through CNN layers reminiscent of AlexNet, followed by fully connected layers with dropout to produce 1×3 parameter output

<div align="center">
    <img src="README/Fig S6.svg" width="60%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S6.</strong> Training and validation loss curves of the Parameter PNN
</p>

- **Training Results**
  After 200 epochs, the Parameter PNN shows significant overfitting with substantial training-validation loss gap, as shown in Fig. S6.
- **Performance Comparison**
  - Parameter PNN: Training loss 0.002294, validation loss 0.01651
  - Parameter Transformer: Training loss 0.0007855, validation loss 0.003264
  - The ~5-fold validation loss increase demonstrates severe overfitting tendencies and training instability, leading to unacceptable MetaSR performance degradation as detailed in subsequent analyses

## Appendix S7. Transmittance Transformer: Architecture and Training Results

Following the Parameter Transformer's success, we apply Transformer-Encoder architecture to the Transmittance Prediction Module, termed Transmittance Transformer. See Fig. S7 for the architectural overview.

<div align="center">
    <img src="README/Fig S7.svg" width="70%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S7.</strong> Transmittance Transformer’s architecture.
</p>

- **Architecture**
  - Embedding Layer:Processes 1×32×32 patterns and 1×3 parameters through separate linear projections to 1024 dimensions, concatenates to 2048 dimensions, then maps to unified 4096-dimensional tokens
  - Transformer Encoder: 8-layer encoder with 16-head attention, identical to Parameter Transformer configuration
  - Feedforward Networks: Separate branches mapping encoded features to real and imaginary transmittance components, producing two 1×301 output vectors

<div align="center">
    <img src="README/Fig S8.svg" width="60%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S8.</strong> Training and validation loss curves of the Transmittance Transformer
</p>

- **Training Results**
  After 200 epochs, the model achieved excellent convergence with minimal overfitting for both transmittance components, as shown in Fig. S8.
- **Loss Metrics and Reproducibility**
  - Seed 42: Training loss 0.001346 (real), 0.001350 (imaginary), validation loss 0.001321 (real), 0.001318 (imaginary)
  - Seed 3407: Training loss 0.0008290 (real), 0.0008285 (imaginary), validation loss 0.001114 (real), 0.001118 (imaginary)
  - Consistent with Parameter Transformer methodology, all main-text results use seed 42 for reproducibility, while noting improved performance with seed 3407

## Appendix S8. MetaSR’s Performance Validation

To enable comprehensive comparison with MetaDiffusion and I-P DM, we adapted MetaSR for 26-point spectral sampling while maintaining architectural integrity. Here we detail the implementation modifications required for this configuration.

- **Implementation Adaptations**
  - Dataset Preparation: Downsample 301-point transmittance to 26 points using Python indexing
  - Pattern U-Net: Reduce transmittance input dimension (2×301→2×26); optimize training with learning rate 4e-5, weight decay 4e-7, StepLR gamma 0.80
  - Parameter Transformer: Reduce transmittance input dimension (2×301→2×26); optimize with learning rate 2e-5, weight decay 2e-7, unchanged StepLR schedule
  - Transmittance Transformer: Reduce output dimension (301→26 points), embedding dimension (4096→1024); optimize with learning rate 4e-5, weight decay 2e-7, StepLR step size 8, gamma 0.85

- **Training Results**

**Table S3.** Training and validation losses for the three models.
| Model | Training Loss | Validation Loss |
| :---- | :------------ | :-------------- |
| Pattern U-Net | 0.005398 | 0.005241 |
| Parameter Transformer | 0.0006291 | 0.003960 |
| Transmittance Transformer (real) | 0.0007384 | 0.001229 |
| Transmittance Transformer (imag) | 0.0007351 | 0.001221 |

Comparing 301-point and 26-point configurations reveals model-dependent adaptation patterns: Pattern U-Net shows minimal change with slight validation improvement (0.005335→0.005241), Transmittance Transformer demonstrates consistent improvements across both components, while Parameter Transformer exhibits degraded generalization (validation loss 0.003264→0.003960), indicating sensitivity to reduced spectral resolution.

- **Performance Comparison**

<div align="center">
    <img src="README/Fig S9.svg" width="70%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S9.</strong> Comprehensive performance evaluation of the MetaSR metasurface inverse design framework with 26 data points. (a) Distribution of MAE: kernel density estimation (KDE) curves and histograms for both best and average results, with vertical markers indicating mean values and 95th percentiles. (b) Distribution of MSE: parallel analysis with a consistent visualization scheme.
</p>

Fig. S9 reveals that MetaSR achieves marginally superior average performance at 26 points (MAE 0.02818, MSE 0.006788) compared to 301 points (MAE 0.02846, MSE 0.006970), while best-case and 95th-percentile metrics show slight degradation. This indicates that reduced input dimensionality enhances average robustness but constrains peak performance, primarily attributable to Parameter Transformer's diminished accuracy with reduced-dimensional transmittance inputs.

- **Additional Considerations**
  - Initialization Sensitivity: Changing the random seed from 42 to 3407 in 301-point configuration yields ~5% performance improvement, suggesting potential for enhanced robustness through optimized initialization strategies.
  - Inference Speed: Pattern U-Net without cross-attention matches MetaDiffusion's parameter count and baseline inference time (0.4265s vs 0.43s per sample). With attention optimizations, MetaSR achieves ~3× speedup over baseline, enabling batch processing of 1024 samples while maintaining substantial speed advantage over MetaDiffusion through high-throughput parallel processing via the Law of Large Numbers.

## Appendix S9. MetaSR’s Error Propagation Analysis

To understand how errors propagate through MetaSR's multi-stage pipeline, we systematically analyze the performance sensitivity of the Pattern Generation Module (PGM) and Parameter Inference Module (PIM) under controlled perturbations.

<div align="center">
    <img src="README/Fig S10.svg" width="70%" alt="" style="display: block; margin: 0 auto;">
</div>
<p align="center">
    <strong>Fig. S10.</strong> Comparison of original MetaSR pipeline and error propagation experimental pipeline.
</p>

- **Experimental Setup**
We implement controlled degradation through two mechanisms: (1) pattern corruption via random pixel flipping at preset ratios (0-5%) to simulate varying PGM quality, and (2) PIM degradation through additive Gaussian noise injection into Parameter Transformer outputs, scaled by model loss magnitude (0-5.0×). All other components remain unchanged to isolate individual effects.

**Table S4.** Average MAE under systematic error propagation conditions.
| Pattern (%) | 0.0×   | 0.5×    | 1.0×    | 2.0×    | 5.0×    |
|-------------|--------|---------|---------|---------|---------|
| 0.0         | 0.007012 | 0.02954 | 0.03945 | 0.05165 | 0.07227 |
| 0.5         | 0.009606 | 0.03020 | 0.03991 | 0.05232 | 0.07340 |
| 1.0         | 0.01061  | 0.03078 | 0.03970 | 0.05252 | 0.07351 |
| 2.0         | 0.01275  | 0.03114 | 0.04044 | 0.05229 | 0.07344 |
| 5.0         | 0.01989  | 0.03448 | 0.04290 | 0.05460 | 0.07498 |

- **Key Findings**
  - Pattern corruption shows moderate impact: 5% pixel corruption increases MAE by ~2.8× (0.007012→0.01989) under noise-free conditions
  - Parameter inference dominates error propagation: Increasing parameter noise from 0.0× to 5.0× degrades average MAE by ~10.3× (0.007012→0.07227)
  - Error accumulation follows additive patterns rather than multiplicative scaling, indicating bounded error magnification and inherent robustness
  - To maintain practical MAE ≤ 0.03 while achieving diverse pattern generation, PIM accuracy must exceed current Parameter Transformer baseline through enhanced priors and inference constraints
These findings suggest that while MetaSR demonstrates reasonable robustness to pattern imperfections, parameter inference accuracy remains the critical bottleneck for maintaining high-fidelity inverse design performance.

## License

MIT License

Copyright (c) 2025 Jiakang Lao, Zihan Zhao, and Shuyang Zhou

Advanced Microscopy and Instrumentation Research Center, School of Instrumentation Science and Engineering, Harbin Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.