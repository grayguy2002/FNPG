# FNPG Code Repository

This repository contains the baseline code package associated with the manuscript:

`Fourier Neural Pseudo-log Generator for Well Log Reconstruction and Beyond-record Generation`

The purpose of this package is transparency and figure traceability. The notebooks in this directory already contain executed outputs. In other words, the main computational results can be inspected directly from the embedded notebook outputs without rerunning the code.

## Scope

- This repository is intended to document which manuscript figures correspond to which code files.
- Most result figures in the manuscript are code-generated and can be traced to notebook outputs or companion scripts in this directory.
- A small number of manuscript figures are not model-result figures. Those are schematic or dataset-context figures and are therefore outside the scope of direct code reproduction.

## Repository Structure

- `FNPG_reconstruction_part.ipynb`
  Reconstruction experiments and reconstruction figure outputs.
- `FNPG_generation.ipynb`
  Main generation experiments and associated quantitative analyses.
- `runner_generation_implementary.ipynb`
  Colab runner notebook that calls the generation and baseline scripts.
- `runner_kmax_ablation.ipynb`
  Colab runner notebook for the `kmax` ablation figure.
- `fnpg_train.py`
  FNPG training script.
- `fnpg_generation.py`
  Generation inference and analysis script.
- `fnpg_generation_with_morphology.py`
  Generation analysis including inverse-transform morphology validation.
- `flexlognet_reconstruction_inferencing.py`
  FlexLogNet reconstruction comparison script.
- `baseline_inference_for_r2_4.py`
  Baseline inference for sample-level significance testing.
- `kmax_evaluation.py`
  Metric aggregation and plotting for the `kmax` sensitivity study.
- `standardardization_gcrr.py`
  Data standardization utility referenced by the preprocessing workflow.
- `infer/`
  Kansas cross-basin inference pipeline used for the cross-basin boundary test.
  An example input LAS file is included at `infer/example_data/1045063634.las`.

## Figure-to-File Mapping

The manuscript contains both non-code figures and code-result figures.

### Not Directly Code-Generated

- `Figure 1` (`fig1X2.pdf`)
  Conceptual workflow and network architecture schematic.
- `Figure 2` (`fig3.pdf`)
  Dataset sparsity and spatial-distribution background figure.

These two figures are presentation or background figures rather than direct model-result outputs.

### Code-Generated Result Figures

- `Figure 3` (`fig4.pdf`)
  Reconstruction training/test loss.
  Primary source: `FNPG_reconstruction_part.ipynb`
  Related script: `fnpg_train.py`

- `Figure 4` (`fig5.pdf`)
  Reconstruction data-slice examples.
  Primary source: `FNPG_reconstruction_part.ipynb`

- `Figure 5` (`fig8.pdf`)
  FNPG vs. FlexLogNet reconstruction comparison.
  Primary sources: `FNPG_reconstruction_part.ipynb`, `flexlognet_reconstruction_inferencing.py`

- `Figure 6` (`fig9.pdf`)
  Generation result for well `15/9-15`.
  Primary sources: `FNPG_generation.ipynb`, `fnpg_generation.py`

- `Figure 7` (`kgs_comparison_plot.pdf`)
  Cross-basin Kansas example.
  Primary sources: `infer/kgs_pipeline.py` and the files in `infer/`

- `Figure 8` (`fig6.pdf`)
  With/without augmentation loss comparison.
  Primary source chain: `FNPG_generation.ipynb` and `runner_generation_implementary.ipynb`

- `Figure 9` (`fig7.pdf`)
  Bi-LSTM / ResNet / reduced FNPG / vanilla FNPG comparison.
  Primary source chain: `FNPG_generation.ipynb` and `runner_generation_implementary.ipynb`

- `Figure 10` (`kmax_ablation_experiment_2-32.pdf`)
  `kmax` ablation training dynamics.
  Primary sources: `runner_kmax_ablation.ipynb`, `kmax_evaluation.py`

## Supporting Analysis Outputs Available in Notebook Results

The revised review package also includes code-generated supporting analyses. These are produced by `FNPG_generation.ipynb`, `fnpg_generation.py`, or `fnpg_generation_with_morphology.py`.

- `R1_9_feature_importance.pdf`
- `R1_10_correlation_pearson.pdf`
- `R1_10_correlation_spearman.pdf`
- `R1_11_input_distributions.pdf`
- `R1_11_target_distributions.pdf`
- `R2_4_statistical_significance.pdf`

## Important Note on Publication Figures

The final manuscript figures were assembled into publication-ready PDF files for layout consistency. The notebooks preserve the underlying computational outputs and result traces. Therefore, the notebook displays are the direct code evidence, while the manuscript figure PDFs are the final publication-formatted versions.

## Open Data and Model Weights

This repository contains the code and executed notebook outputs. The data files and model checkpoints required for direct rerunning are openly provided through the following Google Drive folder:

<https://drive.google.com/drive/folders/1tQ86gKoIqj7dkwTWmRDljomcTfvMXVuu?usp=sharing>

The shared folder contains the external resources referenced by the notebooks and scripts, including:

- training and test data tensors
- standardized well data tables
- model checkpoints and trained weights
- intermediate evaluation files used by the analysis notebooks
- Kansas cross-basin example inputs and related assets

In other words, both the code in this repository and the corresponding data/model-weight resources have been opened for transparency and reproducibility.

## External Paths Referenced by the Code

The source files preserve the original Colab/Google Drive paths used in the experiments. Typical examples include:

- `/content/drive/MyDrive/logcompletion/data/Teamdata/complete_data_for_PIModel/`
- `/content/drive/MyDrive/logcompletion/data/Teamdata/standard_wells/`
- `/content/drive/MyDrive/logcompletion/model/`
- `/content/drive/MyDrive/logcompletion/results/`
- `/content/drive/MyDrive/logcompletion/KGS/`

These path strings document the original execution environment. For direct use, readers should obtain the corresponding data and model files from the shared Google Drive folder above and place them into a matching directory structure, or adjust paths locally as needed.

## Suggested Reading Order

For readers who want a concise entry point into the repository, the following inspection order is recommended:

1. `FNPG_reconstruction_part.ipynb`
2. `FNPG_generation.ipynb`
3. `runner_kmax_ablation.ipynb`
4. `infer/README.md`

This covers the main reconstruction, generation, ablation, and cross-basin results discussed in the revised manuscript.
