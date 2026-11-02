# AI4Science 2026 — Conference Report

**Conference:** 2026 International Conference on Artificial Intelligence for Science (AI4Science 2026)  
**Location:** Shenzhen, China  
**Sources:** Official conference handbook (keynote abstracts and program) and photographed presentation slides.

> **Scope:** This report summarizes the three keynote speeches and the ten oral presentations for which photographs were provided. Oral presentations without supporting slides and poster sessions are omitted. The level of detail reflects the available evidence; numerical results are those reported in the presentations, not independently reproduced.

## 1. Executive Summary

The conference highlighted a transition in AI for Science from isolated predictive models toward **integrated scientific workflows**, including multimodal data analysis, AI agents, automated laboratories, and closed-loop materials discovery. The most relevant technical lessons were:

- **Autonomous discovery needs reliable physical execution.** Scientific agents must combine planning and reasoning with experimental constraints, safety mechanisms, and feedback from real equipment.
- **Scientific foundation models require rigorous baselines.** Pretraining, fine-tuning, architectural complexity, and reinforcement learning do not automatically improve scientific prediction or generation.
- **Physical representation matters end to end.** Crystal graph construction, periodic boundaries, symmetry transformations, and readout choices can affect both accuracy and physical consistency.
- **Data and verification budgets are design choices.** For crystal generation, sampling more candidates produced a larger improvement than the tested training interventions.
- **Transfer across scientific domains remains difficult.** Distribution shifts between chemical families and measurement conditions can substantially reduce reliability, motivating domain-specific evaluation and uncertainty assessment.

## 2. Keynote Speeches

### Keynote I — Scientific Embodied AI: From Computational Assistance to Self-Evolving Autonomous Scientific Discovery Loop

**Speaker:** Prof. Yonghong Tian — Peking University

- **Vision:** Move from individual scientific computing tools to end-to-end autonomous discovery systems that connect hypothesis generation, experiment design, physical execution, data analysis, and iterative refinement.
- **Two central obstacles:** (1) understanding specialized scientific protocols with conditional dependencies and strict procedural requirements; and (2) translating probabilistic LLM decisions into safe, reliable actions in physical laboratories. The official abstract emphasizes neuro-symbolic architectures, deterministic physical boundaries, and rule-based constraints as potential safeguards.
- **AI4S LAB:** A domain-general scientific platform combining computational tools, cloud software, scientific AI models, multi-agent coordination, and automated wet-lab infrastructure.
- **BIOMA multi-agent system:** Divides the scientific workflow among a theoretical scientist (hypothesis and prediction), an experiment planner, an execution-oriented agent, and a data analyst. The roles support an iterative hypothesis–experiment–analysis cycle.
- **BioProBench:** A benchmark for biological experimental-protocol understanding and reasoning. Presented evaluations showed that existing LLMs can omit critical steps or mishandle procedural logic; reliable protocol interpretation remains a bottleneck for lab automation.
- **CarbonKylin:** A problem-oriented platform for high-performance carbon materials, integrating material preparation, dispersion, reaction, spinning, online characterization, and compounding, coordinated by specialized agents. The ambition extends toward manufacturing-process optimization and pilot-scale operation.
- **Platform strategy:** The speaker contrasted **domain-general platforms** for reusable workflows with **problem-oriented platforms** tailored to particular scientific and industrial problems. Both require hardware–software–AI coordination.
- **Long-term direction:** Progress from **prediction → inverse design → autonomous discovery**, with systems learning from experimental feedback rather than merely making one-off predictions.

**Takeaway:** The primary challenge is no longer only generating scientific suggestions; it is developing trustworthy systems that can act on them, validate outcomes, and continually improve. The photographed slides introduce platform architectures and demonstrations, but do not establish a single quantitative end-to-end performance figure for the full autonomous loop.

### Keynote II — AI Psychology: A New Discipline for Understanding and Cultivating Artificial Minds

**Speaker:** Prof. Dapeng Oliver Wu — City University of Hong Kong

- **Motivation:** AI agents increasingly learn, reason, interact, and perform goal-directed tasks. The speaker argued that observable input–output behavior alone may be insufficient to characterize their behavior.
- **Conceptual framework:** Treat an AI agent's *latent mental state* as an unobserved variable related to goals, memory, affect, and intentions, with observable behavior as its output.
- **Three core questions:** (1) What behavior follows from a specified latent state? (2) What latent state can be inferred from an observed behavior? (3) How can an agent be taught or guided to achieve a goal?
- **Proposed research taxonomy:** Eight sub-disciplines: abnormal, clinical, developmental, cognitive, social, personality, criminal, and educational AI psychology.
- **Potential significance:** The framing may provide useful language for agent interpretability, behavior prediction, human–AI interaction, and multi-agent governance.

**Takeaway:** This was mainly a **conceptual research proposal**, not a quantitative benchmark. Using psychological terminology for AI does not, by itself, establish that present AI agents have subjective consciousness or human emotions.

### Keynote III — LLM Innovation with Chinese Characteristics

**Speaker:** Prof. Haizhou Li — The Chinese University of Hong Kong, Shenzhen

- **Innovation under constraints:** The official abstract argued that restrictions on advanced chips and access to closed-source LLMs are encouraging emphasis on computational efficiency, open-source models, and application-driven development rather than scaling alone.
- **Hardware–software co-optimization:** A photographed slide presented full-parameter post-training of a reported 1.6T-parameter DeepSeek-family model on Huawei Ascend SuperPOD 910C infrastructure. The slide reported a rise in Model FLOPs Utilization (MFU) from **11.67% to 34.22%**, described as approximately **2.93×** improvement following optimizations.
- **Policy and ecosystem comparison:** The speaker presented the United States as innovation-first, the European Union as rights- and regulation-oriented, and China as development-oriented. These are the speaker's analytical categories, not a neutral or comprehensive ranking of the three systems.
- **Broader theme:** Development of scientific and general-purpose LLMs depends on access to compute, software efficiency, open model ecosystems, deployment opportunities, and public acceptance.

**Takeaway:** Efficient implementation and infrastructure co-design can be as strategically important as model size. The slides did not provide an independent evaluation of the reported hardware optimization.

## 3. Selected Oral Presentations

### Oral 2 — A Spectral–Physical Manifold for Scalable Knowledge Transfer in Partial Differential Equation Families

**Speaker:** Zhipeng Ge — Soochow University

- Proposed **Graph Diffusion PINNs (GD-PINNs)** to transfer reusable knowledge across related PDE problems, such as CFD simulations with changing geometries and operating conditions.
- Used spectral–physical task representations, encoder inheritance, and local adaptation to reduce redundant training; coarse target observations and physical constraints help refine the predicted fields.
- Identified limitations around flow-regime changes, bifurcations, time-dependent problems, and zero-shot transfer. Multiple local manifolds and dynamic task representations were proposed as future directions.

**Takeaway:** Physics-aware transfer learning could reduce repeated simulation and surrogate-model training costs, although quantitative improvements were not visible in the supplied slides.

### Oral 3 — Tri-Modal Alignment with Modality-Expert Routing for Spot-Level Tumor Detection

**Speaker:** Tianyi Chen — City University of Hong Kong

- Introduced **TriMoE-ST**, a tri-modal alignment and modality-expert routing approach for spatial spot-level tumor detection.
- Only the title slide was available; the actual data modalities, model implementation, and experimental results could not be established.

### Oral 5 — AI4Science at Synchrotron Radiation Facilities: Uncertainty-Aware Multimodal Learning

**Speaker:** Melaku Lake Tegegne — Institute of High Energy Physics, Chinese Academy of Sciences

- Motivated **multimodal materials learning** by combining complementary characterization and composition data rather than relying on a single input modality.
- Illustrated integration of **XANES spectra, XRD patterns, and chemical composition**, with Materials Project information as part of the materials-data context.
- Highlighted uncertainty-aware learning as a research direction for materials characterization and prediction; the photographs did not show the specific uncertainty method or quantitative results.

**Takeaway:** The integration of experimental spectra, crystal structure information, and composition is promising for materials informatics, provided data alignment and uncertainty are carefully evaluated.

### Oral 9 — PromptBio: An Agentic Platform for End-to-End Computational Biomedical Research

**Speaker:** Minzhe Zhang — PromptBio Inc.

- Presented an AI-agent platform automating the workflow **natural-language prompt → planning → data query → analysis → modeling → insight generation**.
- Demonstrated intended use cases including multi-omics analysis, disease-risk prediction, biomarker and drug-target discovery, and patient screening/stratification.
- Presented **PromptBio-Bench**, an end-to-end bioinformatics benchmark, and **BiomniBench**, a process-level evaluation covering tasks such as statistical testing, differential expression, and pathway analysis.
- The presented end-to-end comparison showed **80.7% overall accuracy** for the PromptGenie system versus **75.8%** for Biomni and **71.7%** for Stella; PromptGenie also had the lowest reported average latency (**2.0 minutes**) in that comparison. These are speaker-reported, benchmark-specific figures.

**Takeaway:** Scientific agents can orchestrate multiple analysis tools into repeatable workflows, but process correctness, reproducibility, and transparent validation matter as much as completion rate.

### Oral 13 — Self-Play with Variational Problem Synthesis in Reinforcement Learning with Verifiable Rewards

**Speaker:** Denis Fedyanin — HSE University, Russia

- Proposed generating new formal mathematical proof tasks **from failed attempts** of a base proof generator, without a teacher model or reference proof.
- Converted structural failures into partial Lean proofs with missing steps, and linear failures into prefix-completion tasks.
- Used a ByT5-small tactic encoder and diversified breadth-first search, followed by a training loop involving verified trajectories, LoRA supervised fine-tuning, and MaxRL-LoRA.
- Discussed Lean as a source of objective, machine-checkable feedback for theorem-proving agents; quantitative improvements for the proposed method were not visible in the photographs.

**Takeaway:** Failure mining and verifiable rewards provide a possible route to improving scientific reasoning systems without relying exclusively on human-authored examples.

### Oral 14 — Learning Transition-State-Informed Reaction Representations

**Speaker:** Chaolong Wang — Shenzhen Loop Area Institute

- Presented a benchmark for reaction-barrier prediction across **Grambow Unified (11,961 reactions), RGD1 v9 Core (93,815), DORTS (8,142), and Cyclo-23-TS (5,249)**, with different barrier-energy definitions.
- Compared **SchNet-1M, PaiNN, DimeNet++, GemNet-T, ClofNet, and MACE**, using MAE, MAPE, and additional reported metrics; the protocol included an unseen-reaction-family test.

**Takeaway:** Generalization across reaction families is an important test of chemical representation learning. No model ranking or performance improvement can be concluded from the single photographed benchmark-setup slide.

### Oral 15 — Cross-Site Carbon Flux Forecasting with a Pretrained Time-Series Model: An Input-Matched Evaluation

**Speaker:** Paul Ezennolim — University of Sheffield

- Compared pretrained **TEMPO-80M** (GPT-2-based time-series model) with four task-trained baselines: Transformer, LSTM, Random Forest, and XGBoost.
- Used identical univariate inputs and a shared training pool of **54 sites**, selected from 274 sites, to make the model comparison more controlled.
- TEMPO was competitive in zero-shot use and narrowly ahead on pooled/macro Nash–Sutcliffe Efficiency after fine-tuning, but the pooled NSE advantage was only about **0.001–0.003**, partly attributed to normalization. **Random Forest retained the best MAE.**
- Limitations included single unseeded neural-model runs and insufficient evidence for statistical significance; possible low-data pretraining benefits remained tentative.

**Takeaway:** Foundation-model claims should be tested against strong, input-matched classical baselines rather than inferred from model scale or pretraining alone.

### Oral 16 — Is Fine-Tuning Worth It? Reusing a Pretrained AI Model to Predict the Stability of New Materials

**Speaker:** Faizan Ahmed — King Fahd University of Petroleum and Minerals

- Audited a pretrained **ALIGNN** crystal GNN for predicting **formation energy per atom**, a stability-relevant regression target rather than direct stable/unstable classification.
- Compared oxide and oxygen-free nitride families. The pretrained model's zero-shot test MAE was **0.0342 eV/atom for oxides** versus **0.0695 eV/atom for nitrides** (**2.03×** higher for nitrides), illustrating family-dependent domain shift.
- Tested partial fine-tuning of the final graph-convolution block and output head (about 330,000 of 4.03 million parameters), varying training sample counts, optimization protocols, and seeds.
- **All 36 tested condition means were worse than their corresponding zero-shot baselines**; all 60 individual runs of the canonical protocol also finished above the zero-shot error. Some individual structures still benefited, so the effect was not uniform at the sample level.
- Despite unsuccessful adaptation, **pretrained checkpoint reuse greatly outperformed training from scratch**. For example, with 50 nitride samples, MAE was **0.072 eV/atom** for pretrained-plus-fine-tuned versus **0.691 eV/atom** from scratch.
- Frozen representations strongly distinguished the two material families (reported family-label AUC **0.9994**), indicating substantial representational separation.

**Takeaway:** Keep the unchanged pretrained checkpoint as an explicit baseline. The evidence is limited to one checkpoint, one property, two chemical families, and three partial fine-tuning protocols; it does **not** imply that all forms of fine-tuning are ineffective.

### Oral 17 — EquiCap: Auditing Equivariance, Periodicity, and Hierarchical Readouts in Crystal Graph Property Prediction

**Speaker:** Eddah K. Sure — Shanghai University

- Introduced **EquiCap**, a controlled **3 × 3** comparison crossing CGCNN-style, PaiNN-inspired scalar–vector, and composition-only backbones with attention, capsule routing, and mean/max readouts.
- Evaluated formation-energy prediction on **45,231 DFT-relaxed MP-20 crystals**, controlling model-design choices and conducting three seeded runs per configuration.
- The best model, **CGCNN-style with capsule readout**, reached **29.2 meV/atom MAE**. The best composition-only baseline achieved **34.4 meV/atom**; the geometry-related gap was **5.2 meV/atom**, below the talk's 10 meV/atom threshold for a meaningful architectural claim.
- A **complete-pipeline symmetry audit** examined rotation, reflection, translation, atom permutation, periodic image relabeling, and supercell replication. With fixed graph edges, differences were near numerical precision; rebuilding edges after transformations could introduce substantial prediction drift (up to about **0.075 eV/atom** for a tested PaiNN-style rotation case).
- Explained why hard top-k neighbor selection and neighbor caps can make graph construction discontinuous. The slide reported **39.6%** of atoms had near-tied 12th and 13th neighbors under a tight distance criterion; 0.68% reached a 64-neighbor cap.
- Argued that hierarchical capsule representations need **routing diagnostics** (e.g., entropy, effective capsule count, coupling mass), rather than interpretability claims based on accuracy alone.

**Takeaway:** Physical symmetry should be audited at the **graph construction + model + readout** level, not inferred from a supposedly equivariant layer. Matched composition baselines are essential when assessing the actual value of geometric information.

### Oral 18 — Curriculum Learning Beats Data Composition and GRPO for LLM-Driven Crystal Structure Generation

**Speaker:** Shehroz Ahmad Shoaib — King Fahd University of Petroleum and Minerals

- Studied crystal structure generation from **chemical composition and space group**, using **Qwen2.5-7B-Instruct + LoRA** to output CIF files, with `StructureMatcher` used to assess structural matches.
- Compared a simpler **MP-20** dataset with a more challenging, chronologically split **MPTS-52** dataset; the reported setup used approximately **24,000 training crystals** and **4,500 optimizer steps**.
- On MPTS-52, **best-of-10** structure-match rates were **30.1%** for standard SFT, **30.1%** for data composition, **30.5%** for a short MP-20 curriculum warm-up, **33.6%** with higher LoRA rank (128), and **25.4% / 25.0%** for discrete/continuous-reward GRPO, respectively.
- The dominant improvement came from **inference-time sampling**: generating ten CIF candidates instead of one increased matching success by **11.6 percentage points** for the same baseline.
- Increasing LoRA rank from 32 to 128 added **3.6 percentage points**; a short 500-step curriculum warm-up added **0.4 points**, while longer warm-ups harmed performance.
- **GRPO underperformed continued SFT**: over the compared additional training steps, the strongest GRPO configuration improved its starting checkpoint by **1.1 points**, whereas continued SFT gained **5.8 points**. The GRPO approach also underperformed across all seven tested crystal systems.
- Published diffusion/LLM results shown for context used different conditioning and sampling settings; they were **not** strictly head-to-head comparisons.

**Takeaway:** For the tested LLM generator, increasing the candidate/verification budget and maintaining a strong SFT baseline was more valuable than the studied RL intervention. An external verifier can be an effective **filter** without necessarily providing a useful training **teacher**.

## 4. Cross-Cutting Conclusions

1. **Scientific automation is becoming a systems problem.** The scientific agent talks emphasized integrated data, models, tools, and laboratories, with reliability of real-world execution as a central unresolved issue.
2. **Model sophistication is not sufficient evidence of scientific value.** Across carbon-flux forecasting, ALIGNN transfer, EquiCap, and LLM crystal generation, simple or unchanged baselines remained highly competitive or superior in important settings.
3. **Generalization must be measured explicitly.** Chemical-family shift, unseen reaction families, changing PDE conditions, and heterogeneous characterization modalities are common obstacles to scientific AI deployment.
4. **Verification and physical consistency deserve dedicated evaluation.** Important checks include formal proof validity, scientific protocol compliance, crystal matching, graph symmetry tests, and safe experimental actuation; these are distinct from generic prediction accuracy.
5. **Allocate computation according to measured benefit.** Crystal generation showed a larger return from candidate sampling than from several training changes. Elsewhere, increasing model capacity or continued SFT outperformed tested alternatives.

## 5. Implications for Future Materials-AI Research

- **Property prediction:** Benchmark a pretrained model *without adaptation* before fine-tuning; evaluate chemical-family and distribution shifts independently, and include classical/compact baselines where appropriate.
- **Crystal representation learning:** Audit invariance under periodic-image relabeling, lattice changes, supercell replication, and geometric transformations. Consider graph construction and readout part of the physical model, not just preprocessing.
- **Generative materials discovery:** Evaluate *validity, uniqueness, structural match, physical plausibility, and verification cost* together. Compare stronger SFT, candidate sampling, and RL under matched budgets.
- **Multimodal materials informatics:** Explore complementary inputs such as composition, simulated crystal structures, XRD, and XANES, with clear treatment of missing modalities and predictive uncertainty.
- **Closed-loop discovery:** Build incrementally from validated prediction and candidate ranking toward experiment planning, physical execution, and feedback-driven optimization; introduce safety and domain constraints before automating irreversible operations.

## 6. Evidence and Reporting Notes

- **Official source:** *AI4Science 2026 Conference Handbook*, including the conference overview, full oral program, and abstracts of Keynotes I–III.
- **Slide source:** Photographs supplied for Oral Presentations 2, 3, 5, 9, 13, 14, 15, 16, 17, and 18, and the three keynotes.
- **Scope limitation:** Presentations without provided slides have intentionally been omitted. One-slide presentations are described only at the level supported by those slides. Reported numbers were taken from photographed results and have not been independently replicated.
