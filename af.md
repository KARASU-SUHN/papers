
# TIM SEM Segmentation & 2D→3D Microstructure Generation — Reference List



| Category | Reference (short) | Paper | Code | Open data / weights | Notes (why it’s relevant) |
|---|---|---|---|---|---|
| Microstructure segmentation (DL) | Durmaz et al., *Nat. Communications* (2021) — deep learning semantic segmentation / complex microstructure inference | https://www.nature.com/articles/s41467-021-26565-5 | — | — | Useful background on DL segmentation for complex microstructures; code/data not clearly public. |
| Microstructure segmentation (DL, transfer learning) | Stuckner et al., *npj Computational Materials* (2022) — MicroNet (pretrained encoders for microscopy segmentation) | https://www.nature.com/articles/s41524-022-00878-5 | https://github.com/nasa/pretrained-microscopy-models | — | Pretraining can reduce the amount of labeled SEM needed for good segmentation. |
| Microstructure segmentation (DL, generalization) | Taufique et al., *npj Computational Materials* (2025) — generalizable microstructure segmentation with minimal ground truth | https://www.nature.com/articles/s41524-025-01801-4 | https://github.com/nasa/pretrained-microscopy-models | — | Focuses on label efficiency / minimal ground truth for segmentation. |
| Microstructure segmentation (U-Net) | Bangaru et al., *Cement and Concrete Composites* (2022) — U-Net for SEM segmentation | https://www.sciencedirect.com/science/article/abs/pii/S0926580522004721 | — | — | A representative “U-Net on SEM micrographs” pipeline (augmentation + supervised masks). |
| Microstructure segmentation (U-Net variant) | Yang et al., *Coatings* (2025) — enhanced U-Net for SEM segmentation (copper microstructures) | https://www.mdpi.com/2079-6412/15/8/969 | — | — | Open-access example of U-Net variants + loss/architecture tweaks for SEM. |
| Instance segmentation (SEM particles) | Rettenberger et al., *npj Computational Materials* (2024) — uncertainty-aware particle segmentation for SEM | https://www.nature.com/articles/s41524-024-01302-w | https://github.com/lrettenberger/Uncertainty-Aware-Particle-Segmentation-for-SEM | https://osf.io/f2y8w/ (data zips); https://github.com/lrettenberger/Uncertainty-Aware-Particle-Segmentation-for-SEM-Data | Practical SEM-focused pipeline + open data; uncertainty can help at hard boundaries (like substrate–TIM contact). |
| Instance segmentation (EM particles, uncertainty) | Yildirim & Cole, *J. Chem. Inf. Model.* (2021) — Bayesian particle instance segmentation (BPartIS) | https://pubs.acs.org/doi/10.1021/acs.jcim.0c01455 | https://github.com/by256/bpartis | — | Particle-style instance segmentation + uncertainty; relevant if TIM phase “particles/ellipses” need instance-level masks. |
| Label-efficient segmentation | Gerçek et al., arXiv (2025) — data-efficient U-Net for carbide segmentation in SEM | https://arxiv.org/abs/2511.11485 | — (models/scripts may be inside the dataset record) | https://rodare.hzdr.de/record/4124 | Example of *very* small labeled set training (with released training data + models). |
| Promptable / foundation segmentation (microscopy) | *micro-sam* (“μSAM”), *Nature Methods* (2025) — SAM adapted for microscopy | https://www.nature.com/articles/s41592-024-02580-4 | https://github.com/computational-cell-analytics/micro-sam/ | https://doi.org/10.5281/zenodo.14036956 ; https://doi.org/10.5281/zenodo.14037020 | Good for interactive refinement at difficult interfaces (scribbles/points/boxes), then export masks for supervised finetuning. |
| Promptable / foundation segmentation (medical, transferable ideas) | Ma et al., *Nat. Communications* (2024) — MedSAM | https://www.nature.com/articles/s41467-024-44824-z | https://github.com/bowang-lab/MedSAM | — | Not SEM-specific, but useful as a template for “SAM + finetune” workflows and prompting-based QC. |
| Promptable / foundation segmentation (microscopy, LLM-guided) | Li et al., arXiv (2025) — uLLSAM (SAM + multimodal LLM for microscopy) | https://arxiv.org/pdf/2505.10769 | https://github.com/ieellee/uLLSAM | — | Research direction for cross-domain microscopy segmentation; may be overkill but interesting. |
| 2D→3D microstructure generation | Kench & Cooper, arXiv (2021) — SliceGAN | https://arxiv.org/abs/2102.07708 | https://github.com/stke9/SliceGAN | — | Generates statistically-equivalent 3D volumes from 2D micrographs (relevant for building 3D TIM geometry from SEM). |
| 2D→3D microstructure dataset | Kench et al., *Scientific Data* (2022) — MicroLib (SliceGAN-applied library) | https://www.nature.com/articles/s41597-022-01744-1 | https://github.com/stke9/SliceGAN | https://zenodo.org/records/7118559 | Ready-to-use 3D volumes + 2D sources; good for benchmarking/evaluation protocols. |
| 2D→3D generation (large volume) | Sugiura et al., *Journal of Imaging* (2023) — Big-Volume SliceGAN | https://www.mdpi.com/2313-433X/9/5/90 | — | — | Architectural tweaks for larger/high-res 3D generation (helpful when targeting large voxel grids). |
| 2D→3D generation (optimization-guided) | MicroLad, arXiv (2025) — 2D-to-3D reconstruction & property optimization | https://arxiv.org/html/2508.20138v1 | — | — | Shows a newer 2D→3D direction, linking generation to property objectives. |
| 3D reconstruction (non-DL; SEM-informed) | Long et al., *Ceramics International* (2021) — SEM-informed numerical 3D reconstruction for porous media/TBC | https://www.sciencedirect.com/science/article/abs/pii/S0272884221010622 | — | — | Not deep learning, but relevant “SEM → 3D reconstruction → thermal conductivity” workflow. |
| TIM/Interface-related (property learning) | Kim et al., *Materials Today Communications* (2025) — predicting thermal conductivity from SEM images | https://www.sciencedirect.com/science/article/pii/S0264127525010822 | — | — | Not segmentation-focused, but close to your end goal (SEM → thermal conductivity proxy). |
| Interface/Contact resistance (related) | Zhou et al., *npj Materials Degradation* (2025) — DL prediction/interpretability for thermal contact resistance | https://www.nature.com/articles/s44172-025-00508-0 | — | — | Directly related to *contact* and thermal resistance across interfaces (useful for feature/metric ideas). |




# References — TIM SEM Segmentation / 2D→3D Reconstruction / Thermal Modeling

## A) Deep learning segmentation for materials / SEM microstructures

| ID | Category | Paper (Authors) | Year | Venue | Task / Data | Why it’s relevant to TIM interface segmentation |
|---:|---|---|---:|---|---|---|
| S1 | Semantic segmentation (materials) | Durmaz et al. | 2021 | Nature Communications | Materials microstructure semantic segmentation (U-Net style) | Strong “materials microstructure” precedent; useful for expected annotation scale & workflow |
| S2 | Semantic segmentation (SEM) | Bangaru et al. | 2022 | Cement & Concrete Composites | U-Net segmentation on SEM-like microstructure images | Practical example of U-Net pipeline on SEM micrographs |
| S3 | Semantic segmentation (SEM) | Yang et al. | 2025 | Coatings | SEM microstructure segmentation with U-Net family | Another SEM segmentation workflow to compare training setup/augmentation |

## B) Transfer learning / data-efficient segmentation (important when labels are limited)

| ID | Category | Paper (Authors) | Year | Venue | Task / Data | Why it’s relevant |
|---:|---|---:|---:|---|---|---|
| T1 | Microscopy pretraining | Stuckner et al. (“MicroNet”) | 2022 | npj Computational Materials / NASA TM | Large microscopy dataset + pretrained encoders | Helps reduce label needs; better transfer than ImageNet for microscopy/SEM-like textures |
| T2 | Minimal ground-truth / generalization | Taufique et al. | 2025 | npj Computational Materials | Generalizable segmentation with minimal GT (cross-modality idea) | Strategies to cut annotation cost and improve robustness |
| T3 | Data-efficient U-Net | Gerçek et al. | 2025 | arXiv | Data-efficient U-Net for SEM microstructure | Directly aligned with “small-data SEM segmentation” constraint |

## C) Instance segmentation / particle-focused segmentation (useful if filler particles touch/overlap)

| ID | Category | Paper (Authors) | Year | Venue | Task / Data | Why it’s relevant to packed spheres/ellipses + resin |
|---:|---|---:|---:|---|---|---|
| I1 | Particle segmentation + uncertainty | Rettenberger et al. | 2024 | npj Computational Materials | Particle segmentation in SEM with uncertainty awareness | Useful when boundaries are ambiguous at substrate–TIM contact |
| I2 | Uncertainty / Bayesian segmentation | Yildirim et al. | 2021 | ACS JCIM | Bayesian instance segmentation in EM images | Concepts for confidence maps + targeted relabeling near contact zone |

## D) Foundation models / prompt-based segmentation (bootstrapping labels faster)

| ID | Category | Paper / Model | Year | Venue | What it is | Why it’s relevant |
|---:|---|---|---:|---|---|---|
| F1 | SAM for microscopy | μSAM | 2025 | Nature Methods | Segment Anything adapted to microscopy | Rapid interactive mask creation; good for speeding up annotation |
| F2 | Domain-specific SAM | MedSAM | 2024 | Nature Communications | SAM adapted to medical images | Evidence that domain-tuning improves performance with fewer labels (workflow transferable) |

## E) 2D SEM → 3D microstructure generation (for building 3D volumes from limited 2D)

| ID | Category | Paper / Method | Year | Venue | What it does | Why it’s relevant to your 2D→3D plan |
|---:|---|---|---:|---|---|---|
| G1 | 2D→3D generative | SliceGAN | 2021 | arXiv | Generates 3D microstructure from a representative 2D slice | Popular baseline for statistical 3D reconstruction from 2D |
| G2 | Dataset / evaluation | MicroLib | 2022 | Scientific Data | Benchmark dataset + SliceGAN usage across microstructures | Helps define “statistical fidelity” and evaluation practices |
| G3 | Large-volume generation | Big-Volume SliceGAN | 2023 | (paper/preprint) | Extends SliceGAN toward larger 3D volumes | Useful when you need simulation-sized volumes |

## F) TIM-adjacent thermal modeling from microstructure (SEM→property / sim↔real bridging)

| ID | Category | Paper (Authors) | Year | Venue | What it does | Why it’s relevant |
|---:|---|---:|---:|---|---|---|
| M1 | SEM→thermal conductivity | Kim et al. | 2025 | Materials & Design | Predicts composite thermal conductivity from SEM images; sim↔SEM adaptation | Very close to your end-to-end goal (microstructure → thermal behavior) |
| M2 | SEM-informed 3D + thermal | Long et al. | 2021 | Ceramics International | Reconstructs 3D porous structure using SEM-informed morphology + thermal analysis | Similar workflow: (2D/SEM) → (3D) → (thermal simulation) |










