---
title: Data Filtering Pipeline Visualizer
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Data Filtering Pipeline Visualizer

An interactive web-based visualization tool for analyzing and debugging data filtering pipelines used in multimodal LLM pre-training.

## Features

- **Dataset Loading**: Support for JSON, JSONL, and Parquet files
- **Interactive Filtering**: Define filtering rules with absolute or percentile-based thresholds
- **Multiple Filter Operations**: Greater than, less than, equal, between, with AND logic
- **Filter Presets**: Save and load filter configurations for reuse
- **Visualizations**:
  - Distribution plots for all numeric metrics
  - Before/after filtering comparison
  - Correlation matrix with redundancy detection
- **Image Browser**:
  - View retained and filtered-out samples separately
  - Pagination and sorting by any metric
  - Detailed sample view with all metadata and percentile ranks
- **Export**: Export filtered sample IDs or full data as JSON

## Usage

1. **Load Dataset**: Enter the path to your dataset file (JSONL/JSON/Parquet)
2. **Explore Data**: View distributions and statistics in the Overview tab
3. **Define Filters**: Add filtering rules in the sidebar
4. **Apply Filters**: Click "Apply Filters" to see the impact
5. **Browse Results**: Use the Image Browser to inspect retained/filtered samples
6. **Save Presets**: Save filter configurations for later use
7. **Export**: Export filtered IDs or data for downstream processing

## Dataset Format

The tool expects datasets with:
- **Image paths**: Column named `images`, `image`, `image_path`, etc.
- **Unique IDs**: Column named `id`, `uid`, `uuid`, etc.
- **Numeric metrics**: Any numeric columns will be detected for filtering

## Image Source

Images are served from the HuggingFace dataset [XinyuLiu1999/cc3m-filtered](https://huggingface.co/datasets/XinyuLiu1999/cc3m-filtered).

## Filter Types

### Absolute Value Filters
Filter based on exact metric values:
- `quality_score >= 0.8`
- `text_length between 100 and 500`

### Percentile Filters
Filter based on percentile ranks:
- `quality_score >= 90th percentile`
- `blur_score <= 10th percentile`

## Tips

- Use **Record Limit** when loading large datasets for faster exploration
- Check the **Correlation** tab to identify redundant filters
- Use **Percentile** mode for filters that should adapt to different datasets
- Save commonly used filter combinations as **Presets**
- Switch between **Retained** and **Filtered Out** tabs to understand what's being removed
