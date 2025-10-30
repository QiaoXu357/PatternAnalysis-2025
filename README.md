Alzheimer’s Disease Classification using ConvNeXt

Author: Qiao Xu

Student Number: 48382128

Difficulty: Hard (Project #8)

1. Problem Description
This project addresses the binary classification of Alzheimer’s Disease (AD) versus
Cognitively Normal (NC) subjects using MRI images from the ADNI dataset.
The goal was to achieve over 0.8 test accuracy using a ConvNeXt-based deep
convolutional model.
Although the model trained successfully, the final test accuracy reached approximately
0.70, which falls below the target threshold.
The performance gap and its possible causes are discussed in the Error Analysis section.

2. Model Overview
The model is a compact ConvNeXt architecture implemented in PyTorch, following Liu
et al. (2022, "A ConvNet for the 2020s").
It features four hierarchical stages with progressive channel expansion [96, 192, 384,
768], depthwise 7×7 convolutions, GELU activation,
LayerNorm2d normalization, DropPath regularization, and a final linear head for binary
classification.

3. Data Preparation
The dataset was provided pre-split into training and testing partitions (no manual splitting
performed).
Each class (AD vs NC) was roughly balanced (~1:1 ratio). MRI scans were converted
from 3D NIfTI (.nii.gz) volumes into 2D PNG slices of size 224×224 pixels
for compatibility with PyTorch pipelines.

Sample input:

Sample output:

3.1 Split Rationale and Pre-processing Justification
Since the ADNI subset already provided a validated 75/25 train-test split (Estimated by
folder size) with approx. 1:1 positive and negative cases, this configuration was retained
to ensure comparability with prior medical-imaging studies and maintain class balance.
All augmentations were selected to preserve medical realism while improving
generalisation under limited data.
Augmentations included RandomResizedCrop(224), RandomAffine(±10°),
RandomHorizontalFlip(p=0.5), and RandomErasing(p=0.25), following practices
described by Shorten & Khoshgoftaar (2019) and Perez & Wang (2017).

4. Training Configuration
Optimizer: AdamW
Learning Rate: 5×10⁻⁵
Weight Decay: 0.01
Label Smoothing: 0.1
Scheduler: Linear Warmup + Cosine Annealing
Warmup Epochs: 5
Total Epochs: 50
EMA Decay: 0.999
Drop Path Rate: 0.2
TTA: 2 (original + flip)
Gradient Clipping: 1.0

5. Code Structure
All modules required are stored in adni, with cli for control

- modules.py — ConvNeXt model definition and blocks
- dataset.py — ADNI dataset loader and augmentation
- train.py — Training loop, EMA, plotting
- eval.py — Evaluation script
- prediction — Predict a given image
- cli.py — CLI for train/prediction commands

Structures:

code folder stores all the code, while AD_NC is the dataset, AD_NC contains 2 folders:
train and test

Structure for the project:

*main.py is useless here, only serves as a backup option if the code breaks

6. Quick Start
Training Command:
python code/cli.py train --data-dir AD_NC --epochs 50 --batch-size 32 --lr 1e-4 --output-
dir model_path.

Evaluation Command:
python code/cli.py eval --data-dir AD_NC --checkpoint best_model.pth --batch-size 32

Prediction Command:
python code/cli.py predict --image path/to/img.jpg --checkpoint out/best_model.pth

* The italicized parts are the parameters to be modified.

7. Results Summary
Test Accuracy: 0.71
Test Loss: ≈0.60

Validation Accuracy (best): ≈0.78
GPU: RTX 3060 (6 GB)
Operating System: Windows 11
CUDA: 12.8
PyTorch: 2.9.0 (cu128)
Training Time: ≈3 hours

8. Example Input and Output
Input: A T1-weighted MRI slice (224×224, float32 tensor normalized to [–1, 1])
Output: Dictionary containing class probabilities and predicted label.

9. Discussion and Error Analysis
Although convergence was stable and data balanced, the test accuracy plateaued at ~70%.
Likely causes include limited sample diversity, loss of 3D spatial context in 2D slicing,
and potential over-regularisation (DropPath 0.2 + Label Smoothing 0.1).
Future improvements could include integrating 3D ConvNeXt or ViT models, fine-tuning
on Med3D pretrained weights,
and adopting stratified K-fold validation for stability.

10. Reproducibility Details
Experiments were conducted on Windows 11, CUDA 12.8, and PyTorch 2.9.0 (cu128).
Random seeds were fixed to ensure deterministic results:
torch.manual_seed(42), numpy.random.seed(42).

11. References
1. Liu, Z. et al. (2022). "A ConvNet for the 2020s." arXiv:2201.03545.
2. Shorten, C., & Khoshgoftaar, T. M. (2019). "A Survey on Image Data Augmentation
for Deep Learning." Journal of Big Data, 6(60).
3. Perez, L., & Wang, J. (2017). "The Effectiveness of Data Augmentation in Image
Classification using Deep Learning." arXiv:1712.04621.
4. Zhong, Z. et al. (2020). "Random Erasing Data Augmentation." AAAI Conference on
Artificial Intelligence.
5. Chandra, S. (2025). COMP3710 Pattern Analysis v1.64 Report, The University of
Queensland.

6. ADNI Consortium (2010–2025). Alzheimer’s Disease Neuroimaging Initiative Dataset.
7. OpenAI (2025). ChatGPT (GPT-5, October 2025). Used for code
debugging/improving, report structuring and documentation support.

12. AI Statement
Parts of this report were prepared with assistance from ChatGPT (GPT-5, OpenAI,
October 2025) to summarize methods, format documentation, and ensure clarity, code
debugging and improving. The code, report and results were developed by myself.

