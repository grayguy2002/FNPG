# KGS FNPG Inference Pipeline

This directory contains the inference pipeline used to apply FNPG to a Kansas Geological Survey (KGS) LAS file. It includes data loading, preprocessing, model inference, inverse normalization, visualization, and metric calculation.

## Directory Layout

```text
infer/
├── README.md
├── kgs_las_reader.py
├── kgs_preprocessor.py
├── kgs_fnpg_inference.py
├── kgs_postprocess.py
├── kgs_pipeline.py
└── example_data/
    └── 1045063634.las
```

## Example Input

The example LAS file included in this repository is:

- Well: `1045063634` (`PRENTICE A-1`)
- Location: Kansas, USA
- Depth range: approximately `1950-4947.5 ft`

## Curve Mapping

The pipeline maps KGS LAS curves to FNPG input and output features as follows:

| FNPG feature | KGS curve | Description |
| --- | --- | --- |
| GR | GR | Gamma ray |
| NPHI | NPHI | Neutron porosity |
| RHOB | RHOB | Bulk density |
| DTC | DELTAT | Compressional slowness |
| CALI | CALI | Caliper |
| RSHA | RT30 | Shallow resistivity |
| RMED | RT60 | Medium resistivity |
| RDEP | RT90 | Deep resistivity |
| SP | SP | Spontaneous potential |

## Quick Start

Run the full pipeline:

```bash
python kgs_pipeline.py
```

Or specify custom paths:

```bash
python kgs_pipeline.py \
    --las_file "path/to/your.las" \
    --model_path "path/to/FNPG_kmax32_best.pt" \
    --output_dir "path/to/output" \
    --kmax 32
```

## Step-by-Step Workflow

### 1. Read and clean the LAS file

```python
from kgs_las_reader import KGSLASReader

reader = KGSLASReader("1045063634.las")
df = reader.read_las()

curve_mapping = {
    "GR": "GR",
    "NPHI": "NPHI",
    "RHOB": "RHOB",
    "DTC": "DELTAT",
    "CALI": "CALI",
    "RSHA": "RT30",
    "RMED": "RT60",
    "RDEP": "RT90",
    "SP": "SP",
}

df_extracted = reader.extract_curves(curve_mapping)
df_clean = reader.clean_data(df_extracted)
df_final = reader.interpolate_missing(df_clean, list(curve_mapping.keys()))
```

### 2. Preprocess the data

```python
from kgs_preprocessor import KGSPreprocessor

preprocessor = KGSPreprocessor(window_size=512, stride=256)
input_tensor, depth_indices, data_normalized = preprocessor.preprocess_for_inference(
    df_final,
    save_stats=True,
    stats_path="kgs_normalization_stats.csv",
)
```

### 3. Run FNPG inference

```python
from kgs_fnpg_inference import FNPGInference, ResultReconstructor

inferencer = FNPGInference("FNPG_kmax32_best.pt")
inferencer.load_model(
    graph_channels=[512, 256, 128, 64, 32],
    batch_size=2,
    fea_litho=["GR", "RHOB", "NPHI", "DTC"],
    modes1=32,
)

predictions = inferencer.predict(input_tensor, batch_size=32)

reconstructor = ResultReconstructor(window_size=512, stride=256)
reconstructed = reconstructor.reconstruct(predictions, depth_indices, total_length)
```

### 4. Inverse-transform predictions

```python
from kgs_postprocess import InverseNormalizer

inverse_normalizer = InverseNormalizer("kgs_normalization_stats.csv")
pred_denormalized = inverse_normalizer.inverse_transform(
    reconstructed,
    ["CALI", "RSHA", "RMED", "RDEP", "SP"],
)
```

### 5. Visualize and evaluate

```python
from kgs_postprocess import ResultVisualizer, calculate_metrics

visualizer = ResultVisualizer()
visualizer.plot_comparison(depth, df_original, df_predicted, features)
```

## Output Files

The pipeline writes the following outputs to the selected output directory:

| File | Description |
| --- | --- |
| `kgs_processed.csv` | Cleaned and interpolated LAS data |
| `kgs_normalization_stats.csv` | Normalization statistics |
| `kgs_input_tensor.pt` | Preprocessed input tensor |
| `kgs_predictions_normalized.csv` | Normalized predictions |
| `kgs_predictions_denormalized.csv` | Inverse-transformed predictions |
| `kgs_evaluation_metrics.csv` | MAE, RMSE, R², and correlation metrics |
| `kgs_comparison_plot.png` | Original vs. predicted comparison plot |
| `kgs_prediction_plot.png` | Prediction-only plot |
| `kgs_statistics_plot.png` | Distribution/statistics comparison plot |

## Preprocessing Notes

### LAS reading

- `lasio` is used to read the LAS file.
- Null markers such as `-999.25` are converted to `NaN`.
- The required curves are extracted using the mapping table above.

### Data cleaning

- Depth intervals with all input features missing are removed.
- Missing values are linearly interpolated where possible.
- Missing-value ratios are summarized during preprocessing.

### Sliding-window inference

- Window size: `512`
- Stride: `256`
- Overlapping windows are merged by averaging overlapping samples.

## Model Configuration

Default FNPG inference settings:

```python
{
    "graph_channels": [512, 256, 128, 64, 32],
    "batch_size": 2,
    "fea_litho": ["GR", "RHOB", "NPHI", "DTC"],
    "head": 1,
    "drop": 0,
    "modes1": 32,
}
```

Default inference batch size:

- `32` for prediction batches

Execution device:

- CUDA if available, otherwise CPU

## Metrics

The pipeline computes the following metrics for each predicted curve:

1. `MAE`
2. `RMSE`
3. `R²`
4. `Pearson correlation`

## Dependencies

Install the required packages with:

```bash
pip install numpy pandas torch lasio matplotlib scipy
```

## Standardization Note

Pipeline behavior depends on the normalization statistics available to the user.

If training-distribution statistics are available, they should be used for the most consistent inference behavior. If they are not available, statistics can be estimated from the KGS well itself, but that may shift the value range relative to the original training environment.

## Practical Notes

- This pipeline is designed for single-well inference from a LAS file.
- The included LAS example is provided as a reference input.
- Paths inside the scripts may still reflect the original local or Colab working environment and can be adjusted as needed.

## Troubleshooting

### Import error for `fnpg_train`

If you see an error such as:

```text
No module named 'fnpg_train'
```

make sure the parent `code/` directory is on the Python path and that `fnpg_train.py` is present in the repository root.

### CUDA out of memory

Reduce the inference batch size, for example:

```bash
python kgs_pipeline.py --inference_batch_size 16
```

### LAS file not found

Verify the LAS path and use an absolute path if necessary.
