"""
Data Filtering Pipeline Visualization Tool for Multimodal LLM Pre-training

This tool provides interactive visualization and filtering capabilities for
large-scale datasets used in multimodal LLM pre-training.
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, Response, redirect
from flask_cors import CORS
from pathlib import Path
from functools import lru_cache
import hashlib

app = Flask(__name__)

# HuggingFace dataset configuration for serving images
HF_DATASET_REPO = "XinyuLiu1999/cc3m-filtered"
HF_DATASET_IMAGE_FOLDER = "images"  # Folder within the dataset containing images
CORS(app, resources={r"/*": {"origins": "*"}})

# Disable strict slashes to avoid redirects
app.url_map.strict_slashes = False

# Global state for loaded dataset
dataset_state = {
    'df': None,
    'file_path': None,
    'numeric_columns': [],
    'percentiles': {},
    'stats': {}
}


def load_jsonl(file_path, limit=None):
    """Load JSONL file into pandas DataFrame."""
    # Ensure limit is an integer if provided
    if limit is not None:
        limit = int(limit)
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def load_json(file_path, limit=None):
    """Load JSON file into pandas DataFrame."""
    # Ensure limit is an integer if provided
    if limit is not None:
        limit = int(limit)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        if limit:
            data = data[:limit]
        return pd.DataFrame(data)
    return pd.DataFrame([data])


def load_parquet(file_path, limit=None):
    """Load Parquet file into pandas DataFrame."""
    # Ensure limit is an integer if provided
    if limit is not None:
        limit = int(limit)
    df = pd.read_parquet(file_path)
    if limit:
        df = df.head(limit)
    return df


def detect_numeric_columns(df):
    """Detect numeric columns suitable for filtering."""
    # Columns to skip - these should never be converted to numeric
    skip_patterns = ['id', 'text', 'caption', 'url', 'path', 'image', 'name',
                     'description', 'title', 'content', 'source', 'exif',
                     'status', 'error', 'message', 'data_source']

    # First, try to convert object columns that might be numeric
    # But skip columns that contain lists, dicts, text, or IDs
    for col in df.columns:
        if df[col].dtype == 'object':
            col_lower = col.lower()

            # Skip columns with names suggesting text/ID content
            if any(pattern in col_lower for pattern in skip_patterns):
                continue

            # Check first non-null value to see if it's a scalar string
            first_valid = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if first_valid is not None:
                # Skip columns with list/dict values
                if isinstance(first_valid, (list, dict)):
                    continue
                # Skip columns where the first value looks like text (contains spaces or letters)
                if isinstance(first_valid, str):
                    # If it contains spaces or non-numeric chars (except . and -), skip it
                    if ' ' in first_valid or not first_valid.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                        continue

            try:
                # Try conversion - only keep if most values convert successfully
                converted = pd.to_numeric(df[col], errors='coerce')
                # Only use conversion if at least 90% of non-null values converted
                original_count = df[col].notna().sum()
                converted_count = converted.notna().sum()
                if original_count > 0 and converted_count / original_count >= 0.9:
                    df[col] = converted
            except:
                pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out ID-like columns and very sparse columns
    excluded_patterns = ['_id', 'index', 'idx']
    filtered_cols = []
    for col in numeric_cols:
        col_lower = col.lower()
        # Also exclude 'id' exactly
        if col_lower == 'id':
            continue
        if not any(pattern in col_lower for pattern in excluded_patterns):
            try:
                # Check if column has reasonable variance
                std_val = df[col].std()
                if pd.notna(std_val) and std_val > 0:
                    filtered_cols.append(col)
            except Exception:
                # Skip columns that cause errors
                pass
    return filtered_cols


def compute_percentiles(df, columns):
    """Precompute percentiles for all numeric columns."""
    percentiles = {}
    percentile_values = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for col in columns:
        try:
            valid_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(valid_data) > 0:
                # Use string keys to avoid JSON serialization issues with mixed int/str keys
                percentiles[col] = {
                    str(p): float(np.percentile(valid_data, p))
                    for p in percentile_values
                }
                percentiles[col]['min'] = float(valid_data.min())
                percentiles[col]['max'] = float(valid_data.max())
                percentiles[col]['mean'] = float(valid_data.mean())
                percentiles[col]['std'] = float(valid_data.std())
        except Exception:
            # Skip columns that cause errors
            pass
    return percentiles


def compute_stats(df, columns):
    """Compute comprehensive statistics for numeric columns."""
    stats = {}
    for col in columns:
        try:
            valid_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(valid_data) > 0:
                stats[col] = {
                    'count': int(len(valid_data)),
                    'missing': int(df[col].isna().sum()),
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'median': float(valid_data.median()),
                    'q1': float(valid_data.quantile(0.25)),
                    'q3': float(valid_data.quantile(0.75))
                }
        except Exception:
            # Skip columns that cause errors
            pass
    return stats


def apply_filters(df, filters):
    """
    Apply filtering rules to the dataset.

    filters: list of dicts with keys:
        - column: column name
        - operator: 'gt', 'gte', 'lt', 'lte', 'eq', 'between'
        - value: threshold value (or [min, max] for 'between')
        - type: 'absolute' or 'percentile'
    """
    mask = pd.Series([True] * len(df), index=df.index)

    for f in filters:
        col = f.get('column')
        op = f.get('operator', 'gte')
        value = f.get('value')
        value_type = f.get('type', 'absolute')

        if col not in df.columns:
            continue

        # Convert percentile to absolute value if needed
        if value_type == 'percentile':
            if op == 'between':
                value = [
                    np.percentile(df[col].dropna(), value[0]),
                    np.percentile(df[col].dropna(), value[1])
                ]
            else:
                value = np.percentile(df[col].dropna(), value)

        # Apply the filter
        col_data = df[col]
        if op == 'gt':
            mask &= col_data > value
        elif op == 'gte':
            mask &= col_data >= value
        elif op == 'lt':
            mask &= col_data < value
        elif op == 'lte':
            mask &= col_data <= value
        elif op == 'eq':
            mask &= col_data == value
        elif op == 'between':
            mask &= (col_data >= value[0]) & (col_data <= value[1])

    return mask


def compute_histogram(data, bins=50):
    """Compute histogram data for a series."""
    valid_data = data.dropna()
    if len(valid_data) == 0:
        return {'bins': [], 'counts': []}

    counts, bin_edges = np.histogram(valid_data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        'bins': bin_centers.tolist(),
        'counts': counts.tolist(),
        'bin_edges': bin_edges.tolist()
    }


def compute_correlation_matrix(df, columns):
    """Compute correlation matrix for numeric columns."""
    subset = df[columns].dropna()
    if len(subset) < 2:
        return {'columns': columns, 'matrix': []}

    corr_matrix = subset.corr().values.tolist()
    return {
        'columns': columns,
        'matrix': corr_matrix
    }


# ============== API Routes ==============

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


@app.route('/')
def index():
    """Serve the main visualization interface."""
    return render_template('index.html')


@app.route('/api/load', methods=['POST'])
def load_dataset():
    """Load a dataset from the filesystem."""
    data = request.json
    file_path = data.get('file_path')
    limit = data.get('limit')  # Optional limit for large files

    if not file_path:
        return jsonify({'error': 'No file path provided'}), 400

    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    try:
        # Determine file type and load
        ext = Path(file_path).suffix.lower()
        if ext == '.jsonl':
            df = load_jsonl(file_path, limit)
        elif ext == '.json':
            df = load_json(file_path, limit)
        elif ext == '.parquet':
            df = load_parquet(file_path, limit)
        else:
            return jsonify({'error': f'Unsupported file format: {ext}'}), 400

        # Detect numeric columns
        numeric_cols = detect_numeric_columns(df)

        # Compute percentiles and stats
        percentiles = compute_percentiles(df, numeric_cols)
        stats = compute_stats(df, numeric_cols)

        # Update global state
        dataset_state['df'] = df
        dataset_state['file_path'] = file_path
        dataset_state['numeric_columns'] = numeric_cols
        dataset_state['percentiles'] = percentiles
        dataset_state['stats'] = stats

        # Detect image path column
        image_column = None
        for col in ['images', 'image', 'image_path', 'img_path', 'path']:
            if col in df.columns:
                image_column = col
                break

        # Detect ID column
        id_column = None
        for col in ['id', 'uid', 'uuid', 'index', 'idx']:
            if col in df.columns:
                id_column = col
                break

        return jsonify({
            'success': True,
            'total_records': len(df),
            'numeric_columns': numeric_cols,
            'all_columns': df.columns.tolist(),
            'image_column': image_column,
            'id_column': id_column,
            'percentiles': percentiles,
            'stats': stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get dataset statistics."""
    if dataset_state['df'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    return jsonify({
        'total_records': len(dataset_state['df']),
        'numeric_columns': dataset_state['numeric_columns'],
        'percentiles': dataset_state['percentiles'],
        'stats': dataset_state['stats']
    })


@app.route('/api/histogram/<column>')
def get_histogram(column):
    """Get histogram data for a specific column."""
    if dataset_state['df'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    if column not in dataset_state['df'].columns:
        return jsonify({'error': f'Column not found: {column}'}), 404

    bins = request.args.get('bins', 50, type=int)
    histogram = compute_histogram(dataset_state['df'][column], bins)

    return jsonify({
        'column': column,
        'histogram': histogram,
        'stats': dataset_state['stats'].get(column, {})
    })


@app.route('/api/correlation')
def get_correlation():
    """Get correlation matrix for numeric columns."""
    if dataset_state['df'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    columns = request.args.getlist('columns')
    if not columns:
        columns = dataset_state['numeric_columns']

    correlation = compute_correlation_matrix(dataset_state['df'], columns)
    return jsonify(correlation)


@app.route('/api/filter', methods=['POST'])
def filter_dataset():
    """Apply filters and return statistics."""
    if dataset_state['df'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    data = request.json
    filters = data.get('filters', [])

    df = dataset_state['df']

    # Apply filters
    mask = apply_filters(df, filters)
    filtered_df = df[mask]

    # Compute before/after stats
    before_stats = {}
    after_stats = {}

    for col in dataset_state['numeric_columns']:
        before_data = pd.to_numeric(df[col], errors='coerce').dropna()
        after_data = pd.to_numeric(filtered_df[col], errors='coerce').dropna()

        before_stats[col] = {
            'count': int(len(before_data)),
            'mean': float(before_data.mean()) if len(before_data) > 0 else 0,
            'std': float(before_data.std()) if len(before_data) > 0 else 0,
            'median': float(before_data.median()) if len(before_data) > 0 else 0,
            'min': float(before_data.min()) if len(before_data) > 0 else 0,
            'max': float(before_data.max()) if len(before_data) > 0 else 0
        }

        after_stats[col] = {
            'count': int(len(after_data)),
            'mean': float(after_data.mean()) if len(after_data) > 0 else 0,
            'std': float(after_data.std()) if len(after_data) > 0 else 0,
            'median': float(after_data.median()) if len(after_data) > 0 else 0,
            'min': float(after_data.min()) if len(after_data) > 0 else 0,
            'max': float(after_data.max()) if len(after_data) > 0 else 0
        }

    # Compute filtered histograms
    histograms = {}
    for col in dataset_state['numeric_columns']:
        histograms[col] = {
            'before': compute_histogram(df[col]),
            'after': compute_histogram(filtered_df[col])
        }

    return jsonify({
        'before_count': len(df),
        'after_count': len(filtered_df),
        'retention_rate': len(filtered_df) / len(df) if len(df) > 0 else 0,
        'before_stats': before_stats,
        'after_stats': after_stats,
        'histograms': histograms,
        'filtered_indices': mask[mask].index.tolist()[:10000]  # Limit for performance
    })


@app.route('/api/images', methods=['POST'])
def get_images():
    """Get paginated image data with optional filters."""
    if dataset_state['df'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    data = request.json
    filters = data.get('filters', [])
    page = data.get('page', 1)
    per_page = data.get('per_page', 50)
    sort_by = data.get('sort_by')
    sort_order = data.get('sort_order', 'asc')

    df = dataset_state['df']

    # Apply filters
    mask = apply_filters(df, filters)
    filtered_df = df[mask].copy()

    # Sort if requested
    if sort_by and sort_by in filtered_df.columns:
        ascending = sort_order == 'asc'
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    # Paginate
    total = len(filtered_df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = filtered_df.iloc[start:end]

    # Extract image data
    images = []
    for idx, row in page_df.iterrows():
        image_data = {'_index': int(idx)}

        # Get all columns
        for col in df.columns:
            val = row[col]
            # Check for NA - need to handle scalar vs array cases
            # pd.isna on a list returns a list, so check isinstance first
            if isinstance(val, (list, dict)):
                image_data[col] = val
            elif pd.isna(val) if np.isscalar(val) or val is None else False:
                image_data[col] = None
            elif isinstance(val, (np.integer, np.floating)):
                image_data[col] = float(val) if isinstance(val, np.floating) else int(val)
            else:
                image_data[col] = str(val)

        images.append(image_data)

    return jsonify({
        'images': images,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page
    })


@app.route('/api/images/filtered-out', methods=['POST'])
def get_filtered_out_images():
    """Get paginated image data for samples that are FILTERED OUT (inverse of filters)."""
    if dataset_state['df'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    data = request.json
    filters = data.get('filters', [])
    page = data.get('page', 1)
    per_page = data.get('per_page', 50)
    sort_by = data.get('sort_by')
    sort_order = data.get('sort_order', 'asc')

    df = dataset_state['df']

    # Apply filters and get the INVERSE (filtered out samples)
    if filters:
        mask = apply_filters(df, filters)
        filtered_out_df = df[~mask].copy()  # Invert the mask to get filtered out samples
    else:
        # If no filters, nothing is filtered out
        filtered_out_df = pd.DataFrame(columns=df.columns)

    # Sort if requested
    if sort_by and sort_by in filtered_out_df.columns and len(filtered_out_df) > 0:
        ascending = sort_order == 'asc'
        filtered_out_df = filtered_out_df.sort_values(by=sort_by, ascending=ascending)

    # Paginate
    total = len(filtered_out_df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = filtered_out_df.iloc[start:end]

    # Extract image data
    images = []
    for idx, row in page_df.iterrows():
        image_data = {'_index': int(idx)}

        # Get all columns
        for col in df.columns:
            val = row[col]
            if isinstance(val, (list, dict)):
                image_data[col] = val
            elif pd.isna(val) if np.isscalar(val) or val is None else False:
                image_data[col] = None
            elif isinstance(val, (np.integer, np.floating)):
                image_data[col] = float(val) if isinstance(val, np.floating) else int(val)
            else:
                image_data[col] = str(val)

        images.append(image_data)

    return jsonify({
        'images': images,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': max(1, (total + per_page - 1) // per_page)
    })


@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """Redirect to HuggingFace dataset URL for the image."""
    # Extract the filename from the path (e.g., "000057623.jpg" from full path)
    filename = os.path.basename(image_path)

    # Construct HuggingFace dataset URL
    # Format: https://huggingface.co/datasets/{repo}/resolve/main/{folder}/{filename}
    hf_url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{HF_DATASET_IMAGE_FOLDER}/{filename}"

    return redirect(hf_url)


@app.route('/api/sample/<int:index>')
def get_sample(index):
    """Get detailed data for a specific sample."""
    if dataset_state['df'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    df = dataset_state['df']

    if index < 0 or index >= len(df):
        return jsonify({'error': 'Index out of range'}), 404

    row = df.iloc[index]
    sample_data = {}

    for col in df.columns:
        val = row[col]
        # Check for NA - need to handle scalar vs array cases
        # pd.isna on a list returns a list, so check isinstance first
        if isinstance(val, (list, dict)):
            sample_data[col] = val
        elif pd.isna(val) if np.isscalar(val) or val is None else False:
            sample_data[col] = None
        elif isinstance(val, (np.integer, np.floating)):
            sample_data[col] = float(val) if isinstance(val, np.floating) else int(val)
        else:
            sample_data[col] = str(val)

    # Add percentile ranks for numeric columns
    percentile_ranks = {}
    for col in dataset_state['numeric_columns']:
        if col in sample_data and sample_data[col] is not None:
            val = sample_data[col]
            col_data = df[col].dropna()
            rank = (col_data < val).sum() / len(col_data) * 100
            percentile_ranks[col] = round(rank, 2)

    sample_data['_percentile_ranks'] = percentile_ranks
    sample_data['_index'] = index

    return jsonify(sample_data)


@app.route('/api/export', methods=['POST'])
def export_filtered():
    """Export filtered dataset indices or save filtered data."""
    if dataset_state['df'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    data = request.json
    filters = data.get('filters', [])
    export_format = data.get('format', 'indices')  # 'indices' or 'json'

    df = dataset_state['df']
    mask = apply_filters(df, filters)
    filtered_df = df[mask]

    if export_format == 'indices':
        # Return just the IDs/indices of filtered samples
        id_col = None
        for col in ['id', 'uid', 'uuid', 'index']:
            if col in filtered_df.columns:
                id_col = col
                break

        if id_col:
            ids = filtered_df[id_col].tolist()
        else:
            ids = filtered_df.index.tolist()

        return jsonify({
            'filtered_count': len(ids),
            'ids': ids
        })

    elif export_format == 'json':
        # Return the filtered data as JSON
        return jsonify({
            'filtered_count': len(filtered_df),
            'data': filtered_df.to_dict(orient='records')
        })

    return jsonify({'error': 'Invalid export format'}), 400


# ============== Filter Presets ==============

# Directory to store filter presets
PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presets')


def ensure_presets_dir():
    """Ensure the presets directory exists."""
    if not os.path.exists(PRESETS_DIR):
        os.makedirs(PRESETS_DIR)


@app.route('/api/presets', methods=['GET'])
def list_presets():
    """List all saved filter presets."""
    ensure_presets_dir()
    presets = []

    for filename in os.listdir(PRESETS_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(PRESETS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    presets.append({
                        'name': filename[:-5],  # Remove .json extension
                        'description': data.get('description', ''),
                        'filter_count': len(data.get('filters', [])),
                        'created_at': data.get('created_at', ''),
                        'dataset_hint': data.get('dataset_hint', '')
                    })
            except:
                pass

    # Sort by name
    presets.sort(key=lambda x: x['name'].lower())
    return jsonify({'presets': presets})


@app.route('/api/presets/<name>', methods=['GET'])
def load_preset(name):
    """Load a specific filter preset."""
    ensure_presets_dir()
    filepath = os.path.join(PRESETS_DIR, f'{name}.json')

    if not os.path.exists(filepath):
        return jsonify({'error': f'Preset not found: {name}'}), 404

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/presets', methods=['POST'])
def save_preset():
    """Save a new filter preset."""
    ensure_presets_dir()
    data = request.json

    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Preset name is required'}), 400

    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in ' -_').strip()
    if not safe_name:
        return jsonify({'error': 'Invalid preset name'}), 400

    filepath = os.path.join(PRESETS_DIR, f'{safe_name}.json')

    from datetime import datetime
    preset_data = {
        'name': safe_name,
        'description': data.get('description', ''),
        'filters': data.get('filters', []),
        'created_at': datetime.now().isoformat(),
        'dataset_hint': dataset_state.get('file_path', '')
    }

    try:
        with open(filepath, 'w') as f:
            json.dump(preset_data, f, indent=2)
        return jsonify({'success': True, 'name': safe_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/presets/<name>', methods=['DELETE'])
def delete_preset(name):
    """Delete a filter preset."""
    ensure_presets_dir()
    filepath = os.path.join(PRESETS_DIR, f'{name}.json')

    if not os.path.exists(filepath):
        return jsonify({'error': f'Preset not found: {name}'}), 404

    try:
        os.remove(filepath)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    import socket

    parser = argparse.ArgumentParser(description='Data Filtering Visualization Tool')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Get the machine's IP address
    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
    except:
        ip_address = '127.0.0.1'

    print("\n" + "="*60)
    print("Data Filtering Pipeline Visualizer")
    print("="*60)
    print(f"\nServer starting on port {args.port}")
    print(f"\nAccess the tool at:")
    print(f"  - Local:   http://127.0.0.1:{args.port}")
    print(f"  - Network: http://{ip_address}:{args.port}")
    print(f"\nHealth check: http://{ip_address}:{args.port}/health")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
