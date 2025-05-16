### Does This Smell the Same? Learning Representations of Olfactory Mixtures Using Inductive Biases

- **POMMix model** predicts olfactory similarity of molecular mixtures using single-molecule data  
- **Graph neural network + attention** used to create mixture-level representations  
- **Cosine similarity head** encodes perceptual distances in learned embedding space  
- **Outperforms baselines** and generalizes well across olfactory datasets  


### Evaluating Universal Interatomic Potentials for Molecular Dynamics of Real-World Minerals

- **Benchmarks 6 UIPs** (e.g., CHGNET, M3GNET, MACE) on 2,400 real mineral structures (AMCSD-MD-2.4K)  
- **ORB and SEVENNET** show highest simulation stability (completion rates >98%)  
- **Density prediction accuracy** varies; MACE, ORB, MATTERSIM perform best (R² > 0.8)  
- **Significant variability** observed across UIPs in MD simulations, indicating need for improvement  


### CrystalGym: A New Benchmark for Materials Discovery Using Reinforcement Learning
- **Introduces CrystalGym**, an open-source RL environment for crystalline material discovery using DFT-based feedback  
- **Benchmarks RL algorithms** (value- and policy-based) on tasks optimizing properties like band gap, bulk modulus, and density  
- **Highlights challenges** in using RL with expensive DFT computations, emphasizing sample efficiency and convergence  
- **Aims to bridge** reinforcement learning and materials science for real-world design applications  



### Compositional Flows for 3D Molecule and Synthesis Pathway Co-design

- **Proposes CGFlow**, combining flow matching and GFlowNets for joint molecule and synthesis design  
- **Models molecule generation** as compositional transitions guided by reward signals  
- **Outperforms baselines** on LIT-PCBA in affinity and diversity metrics  
- **Excels on CrossDocked**, showing high docking scores and synthesis success rates  

### All-Atom Diffusion Transformers: Unified Generative Modelling of Molecules & Materials

- **Proposes ADiT**, a unified latent diffusion framework for jointly generating molecules and materials  
- **Utilizes a shared latent space** via a Transformer-based autoencoder for both periodic and non-periodic systems  
- **Employs a Diffusion Transformer** to sample new latent embeddings, enabling generation of realistic molecules and materials  
- **Achieves state-of-the-art results** on QM9 and MP20 datasets, outperforming molecule- and crystal-specific models  

### MoMa: A Modular Deep Learning Framework for Material Property Prediction

- **Proposes MoMa**, a modular framework addressing diversity and disparity in material property prediction tasks  
- **Trains specialized modules** on various high-resource datasets, then adaptively composes them for downstream tasks  
- **Achieves 14% average improvement** over the strongest baseline across 17 datasets  
- **Demonstrates strong performance** in few-shot and continual learning scenarios, showcasing real-world applicability

### MatBind: Probing the Multimodality of Materials Science with Contrastive Learning

- **Introduces MatBind**, a model aligning four materials science modalities—crystal structures, DOS, pXRD, and text—into a unified embedding space using a hub-and-spoke architecture  
- **Achieves high cross-modal retrieval**, with recall@1 up to 97% for directly aligned modalities and up to 73% for unseen modality pairs  
- **Enables semantic querying across modalities**, facilitating discovery of relationships between different material representations  
- **Demonstrates improved structural recognition**, particularly for perovskite systems, by integrating multiple data modalities

### Towards Extrapolation in Deep Material Property Regression

- **Highlights the challenge** of extrapolating material properties beyond training data distributions, crucial for discovering materials with exceptional properties  
- **Introduces a benchmark** with seven tasks to evaluate extrapolation performance in material property regression (MPR)  
- **Proposes MEX framework**, reframing MPR as a material-property matching problem using cosine similarity and contrastive learning  
- **Demonstrates that MEX** outperforms existing methods in extrapolation tasks, showing potential for advanced material discovery

### Open Materials Generation with Stochastic Interpolants

- **Introduces OMatG**, a generative framework for inorganic crystal design using stochastic interpolants (SI) that unify diffusion and flow-based models  
- **Incorporates equivariant graph representations** and periodic boundary conditions for accurate crystal structure modeling  
- **Couples SI flows** over spatial coordinates and lattice vectors with discrete flow matching for atomic species  
- **Outperforms existing methods** in crystal structure prediction and de novo generation tasks, setting a new state-of-the-art in materials discovery  

### Operating Robotic Laboratories with Large Language Models and Teachable Agents

- **Develops an agentic AI system** integrating Large Language Models (LLMs) to manage sequential laboratory operations in materials design.
- **Introduces teachable agents** that learn from human interactions via in-context learning, storing knowledge in a vector database for long-term memory.
- **Demonstrates adaptability** of AI agents to complex, multi-task workflows, enhancing usability and reproducibility in scientific facilities.
- **Bridges the gap** between advanced automation and user-friendly operation, paving the way for more intelligent scientific laboratories.

### Evaluating Universal Interatomic Potentials for Molecular Dynamics of Real-World Minerals

- **Benchmarks six UIPs** (e.g., CHGNET, M3GNET, MACE) on 2,400 real mineral structures (AMCSD-MD-2.4K)  
- **ORB and SEVENNET** show highest simulation stability (completion rates >98%)  
- **Density prediction accuracy** varies; MACE, ORB, MATTERSIM perform best (R² > 0.8)  
- **Significant variability** observed across UIPs in MD simulations, indicating need for improvement  

### Evaluating Universal Interatomic Potentials for Molecular Dynamics of Real-World Minerals

- **Benchmarks six UIPs** (e.g., CHGNET, M3GNET, MACE) on 2,400 real mineral structures (AMCSD-MD-2.4K)  
- **ORB and SEVENNET** show highest simulation stability (completion rates >98%)  
- **Density prediction accuracy** varies; MACE, ORB, MATTERSIM perform best (R² > 0.8)  
- **Significant variability** observed across UIPs in MD simulations, indicating need for improvement  

### PLaID: Preference-Aligned Language Model for Targeted Inorganic Materials Design

- **Introduces PLaID**, an LLM fine-tuned on Wyckoff-based text representations of crystals for stable crystal generation  
- **Applies Direct Preference Optimization (DPO)** to align model outputs toward thermodynamically stable structures  
- **Encodes symmetry constraints** directly into text, enabling the model to learn structural parameters implicitly  
- **Achieves a 40% higher rate** of generating stable, unique, and novel structures compared to prior methods
- 

### MLIP Arena: Advancing Fairness and Transparency in Machine Learning Interatomic Potentials via an Open, Accessible Benchmark Platform

- **Introduces MLIP Arena**, a benchmark platform evaluating MLIPs on physics awareness, chemical reactivity, stability under extreme conditions, and predictive capabilities for thermodynamic properties.
- **Highlights limitations** of existing benchmarks, such as data leakage and overreliance on error-based metrics tied to specific DFT references.
- **Provides a reproducible framework** to guide MLIP development toward improved predictive accuracy and runtime efficiency while maintaining physical consistency.
- **Offers a Python package and online leaderboard** available at [huggingface.co/spaces/atomind/mlip-arena](https://huggingface.co/spaces/atomind/mlip-arena).

### Towards Fast, Specialized Machine Learning Force Fields: Distilling Foundation Models via Energy Hessians

- **Proposes a distillation method** transferring knowledge from large MLFF foundation models to smaller, faster specialized models using energy Hessians  
- **Trains student models** to match the Hessians of teacher models, ensuring accurate force predictions and energy conservation  
- **Achieves up to 20× speedup** over foundation models while maintaining or exceeding their accuracy  
- **Enables deployment** of efficient, physically consistent MLFFs tailored to specific chemical subsets  

