"""
Data Filtering Pipeline Visualization Tool for Multimodal LLM Pre-training

This tool provides interactive visualization and filtering capabilities for
large-scale datasets used in multimodal LLM pre-training.
"""

import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file, Response
from flask_cors import CORS
from pathlib import Path
import mimetypes
from functools import lru_cache
import hashlib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Disable strict slashes to avoid redirects
app.url_map.strict_slashes = False

# Global state for loaded dataset
dataset_state = {
    'df': None,
    'file_path': None,
    'numeric_columns': [],
    'percentiles': {},
    'stats': {},
    'has_image_bytes': False,       # Whether dataset has binary image data
    'image_bytes_column': None,     # Column name containing binary image data
    'image_name_column': None       # Column name containing image names (for display)
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


def load_laioncoco(parquet_path, jsonl_path, limit=None):
    """Load laioncoco dataset from a parquet file (with binary images) and a JSONL stats file.

    Supports two parquet schemas:

    Schema A (original):
        uid, clean_content (JSON string), clip_score,
        image_buffer_list (list<struct<buffer: binary, image_id: string>>)

    Schema B (flat):
        uid, clip_score, text, images (list<string>),
        image_bytes (list<binary>), and other top-level columns

    The JSONL has one JSON object per line with a __dj__stats__ key containing
    metric dicts where values are single-element lists.

    Returns (df, stats_columns) where:
    - df: merged DataFrame with all data
    - stats_columns: list of column names that should be used for numeric filtering
    """
    if limit is not None:
        limit = int(limit)

    # Load parquet - if path is a directory, collect all parquet files under it
    if os.path.isdir(parquet_path):
        import glob
        parquet_files = sorted(glob.glob(os.path.join(parquet_path, '**', '*.parquet'), recursive=True))
        if not parquet_files:
            raise ValueError(f'No parquet files found under directory: {parquet_path}')
        dfs = []
        remaining = limit
        for pf in parquet_files:
            part = pd.read_parquet(pf)
            if remaining is not None:
                part = part.head(remaining)
                remaining -= len(part)
            dfs.append(part)
            if remaining is not None and remaining <= 0:
                break
        df_parquet = pd.concat(dfs, ignore_index=True)
    else:
        df_parquet = pd.read_parquet(parquet_path)
        if limit:
            df_parquet = df_parquet.head(limit)

    # Load JSONL stats (row-aligned with parquet)
    stats_records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                stats_records.append(json.loads(line))

    # Detect schema: Schema A has image_buffer_list, Schema B has flat columns
    is_schema_a = 'image_buffer_list' in df_parquet.columns

    # Extract binary image data
    image_bytes_list = []
    image_ids = []

    if is_schema_a:
        # Schema A: extract from image_buffer_list (list<struct<buffer, image_id>>)
        for val in df_parquet['image_buffer_list']:
            if val is not None and len(val) > 0:
                item = val[0]
                # Handle both dict-like and struct-like access
                if isinstance(item, dict):
                    image_bytes_list.append(item.get('buffer'))
                    image_ids.append(item.get('image_id', ''))
                else:
                    # PyArrow struct converted to dict by pandas
                    image_bytes_list.append(getattr(item, 'buffer', None) if hasattr(item, 'buffer') else item.get('buffer', None))
                    image_ids.append(getattr(item, 'image_id', '') if hasattr(item, 'image_id') else item.get('image_id', ''))
            else:
                image_bytes_list.append(None)
                image_ids.append('')
    else:
        # Schema B: extract from flat image_bytes column (bytes or list<binary>)
        if 'image_bytes' in df_parquet.columns:
            for val in df_parquet['image_bytes']:
                image_bytes_list.append(_extract_binary_element(val))
        # Use images column for image IDs if available
        if 'images' in df_parquet.columns:
            for val in df_parquet['images']:
                if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                    image_ids.append(val[0])
                elif isinstance(val, str):
                    image_ids.append(val)
                else:
                    image_ids.append('')

    # Build the merged DataFrame
    result = pd.DataFrame()

    # Add uid
    if 'uid' in df_parquet.columns:
        result['uid'] = df_parquet['uid'].values

    if is_schema_a:
        # Schema A: parse clean_content JSON string to extract useful fields
        parsed_fields = []
        if 'clean_content' in df_parquet.columns:
            for val in df_parquet['clean_content']:
                if isinstance(val, str):
                    try:
                        parsed_fields.append(json.loads(val))
                    except json.JSONDecodeError:
                        parsed_fields.append({})
                elif isinstance(val, dict):
                    parsed_fields.append(val)
                else:
                    parsed_fields.append({})

        if parsed_fields:
            content_df = pd.DataFrame(parsed_fields)
            # Select useful fields (skip image_id_list as it's redundant)
            useful_fields = ['text', 'origin_text', 'top_caption_sim', 'data_type',
                             'width', 'height', 'url', 'pwatermark', 'punsafe']
            for field in useful_fields:
                if field in content_df.columns:
                    result[field] = content_df[field].values
    else:
        # Schema B: copy top-level columns directly (skip binary/structural cols)
        skip_cols = {'uid', 'image_bytes', 'images', 'image_buffer_list'}
        for col in df_parquet.columns:
            if col not in skip_cols:
                result[col] = df_parquet[col].values

    # Add clip_score (Schema A needs explicit copy; Schema B already copied above)
    if is_schema_a and 'clip_score' in df_parquet.columns:
        result['clip_score'] = df_parquet['clip_score'].values

    # Add binary image data as a column
    if image_bytes_list:
        result['image_bytes'] = image_bytes_list
    if image_ids:
        result['image_id'] = image_ids

    # Flatten and merge JSONL stats, tracking which columns come from stats
    stats_columns = set()
    if stats_records:
        for i, rec in enumerate(stats_records):
            if i >= len(result):
                break
            dj_stats = rec.get('__dj__stats__', {})
            for metric_name, metric_val in dj_stats.items():
                stats_columns.add(metric_name)
                if metric_name not in result.columns:
                    result[metric_name] = np.nan
                # Values are stored as single-element lists; extract the scalar
                if isinstance(metric_val, list) and len(metric_val) > 0:
                    result.at[i, metric_name] = float(metric_val[0])
                elif isinstance(metric_val, (int, float)):
                    result.at[i, metric_name] = float(metric_val)

    return result, sorted(stats_columns)


def _extract_binary_element(val):
    """Extract raw bytes from a value that may be bytes, or a list/ndarray of bytes.

    Parquet list<binary> columns are loaded as ndarray([b'...']) by pandas.
    This function unwraps that to get the actual bytes.
    Returns bytes if found, None otherwise.
    """
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    # Handle ndarray or list containing bytes (from parquet list<binary>)
    if isinstance(val, (list, np.ndarray)):
        if len(val) > 0:
            inner = val[0]
            if isinstance(inner, (bytes, bytearray)):
                return bytes(inner)
    return None


def _check_col_has_binary(df, col):
    """Check if a column contains binary image data (possibly nested in arrays)."""
    non_null = df[col].dropna()
    if len(non_null) == 0:
        return False
    first_valid = non_null.iloc[0]
    return _extract_binary_element(first_valid) is not None


def detect_image_bytes_column(df):
    """Detect columns containing binary image data in a DataFrame.

    Handles both flat binary columns and parquet list<binary> columns
    (which pandas loads as ndarray of bytes).

    Returns (image_bytes_col, image_name_col) or (None, None) if not found.
    """
    image_bytes_col = None
    image_name_col = None

    # Check for common binary image column names
    bytes_candidates = ['image_bytes', 'img_bytes', 'image_data', 'img_data',
                        'image_binary', 'img_binary', 'bytes']
    for col in bytes_candidates:
        if col in df.columns:
            if _check_col_has_binary(df, col):
                image_bytes_col = col
                break

    if image_bytes_col is None:
        # Fallback: scan all object columns for binary data
        for col in df.columns:
            if df[col].dtype == 'object':
                if _check_col_has_binary(df, col):
                    image_bytes_col = col
                    break

    # If we found an image_bytes column, look for image name column
    if image_bytes_col:
        name_candidates = ['images', 'image', 'image_name', 'img_name',
                           'image_id', 'image_path', 'img_path', 'filename', 'file_name', 'name']
        for col in name_candidates:
            if col in df.columns and col != image_bytes_col:
                image_name_col = col
                break

    return image_bytes_col, image_name_col


def guess_image_mimetype(data):
    """Guess the MIME type of binary image data from its magic bytes."""
    if data[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    elif data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    elif data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    elif data[:4] == b'\x00\x00\x01\x00':
        return 'image/x-icon'
    elif data[:2] == b'BM':
        return 'image/bmp'
    return 'image/jpeg'  # Default fallback


# Sentinel object to signal a value should be skipped during serialization
_SKIP_SENTINEL = object()


def _serialize_value(val):
    """Serialize a DataFrame cell value to a JSON-compatible type.

    Handles numpy arrays (from parquet list<T> columns), bytes, numpy scalars, etc.
    Returns _SKIP_SENTINEL for binary data that should be excluded.
    """
    # Handle numpy arrays (from parquet list<T> columns like list<string>, list<binary>)
    if isinstance(val, np.ndarray):
        # Check if array contains binary data
        if len(val) > 0 and isinstance(val[0], (bytes, bytearray)):
            return _SKIP_SENTINEL
        # Convert to Python list for JSON serialization
        return val.tolist()
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, (bytes, bytearray)):
        return _SKIP_SENTINEL
    # Scalar NA check
    if np.isscalar(val) or val is None:
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(val, (np.integer, np.floating)):
        return float(val) if isinstance(val, np.floating) else int(val)
    return str(val)


def detect_numeric_columns(df):
    """Detect numeric columns suitable for filtering."""
    # Columns to skip - these should never be converted to numeric
    skip_patterns = ['id', 'text', 'caption', 'url', 'path', 'image', 'name',
                     'description', 'title', 'content', 'source', 'exif',
                     'status', 'error', 'message', 'data_source',
                     'bytes', 'binary', 'blob']

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
                # Skip columns with list/dict/bytes/ndarray values
                if isinstance(first_valid, (list, dict, bytes, bytearray, np.ndarray)):
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
    stats_path = data.get('stats_path')  # JSONL stats file (for laioncoco mode)
    dataset_format = data.get('format', 'default')  # 'default' or 'laioncoco'
    limit = data.get('limit')  # Optional limit for large files

    if not file_path:
        return jsonify({'error': 'No file path provided'}), 400

    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {file_path}'}), 404

    if dataset_format == 'laioncoco':
        if not stats_path:
            return jsonify({'error': 'Stats file path is required for LaionCOCO format'}), 400
        if not os.path.exists(stats_path):
            return jsonify({'error': f'Stats file not found: {stats_path}'}), 404

    try:
        # Determine file type and load
        ext = Path(file_path).suffix.lower()
        stats_columns = None  # Track stats-sourced columns for laioncoco
        if dataset_format == 'laioncoco':
            df, stats_columns = load_laioncoco(file_path, stats_path, limit)
        elif ext == '.jsonl':
            df = load_jsonl(file_path, limit)
        elif ext == '.json':
            df = load_json(file_path, limit)
        elif ext == '.parquet':
            df = load_parquet(file_path, limit)
        else:
            return jsonify({'error': f'Unsupported file format: {ext}'}), 400

        # Detect binary image data columns (e.g., from parquet files)
        image_bytes_col, image_name_col = detect_image_bytes_column(df)
        dataset_state['has_image_bytes'] = image_bytes_col is not None
        dataset_state['image_bytes_column'] = image_bytes_col
        dataset_state['image_name_column'] = image_name_col

        # Detect numeric columns
        numeric_cols = detect_numeric_columns(df)

        # For laioncoco, restrict numeric columns to only those from the stats file
        if stats_columns is not None:
            numeric_cols = [c for c in numeric_cols if c in stats_columns]

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
        if image_bytes_col:
            # When we have binary image data, use the name column for display
            # and signal to the frontend that images come from binary data
            image_column = image_name_col or image_bytes_col
        else:
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

        # Build list of columns to expose (exclude binary data columns)
        exposed_columns = [col for col in df.columns.tolist()
                           if col != image_bytes_col]

        return jsonify({
            'success': True,
            'total_records': len(df),
            'numeric_columns': numeric_cols,
            'all_columns': exposed_columns,
            'image_column': image_column,
            'id_column': id_column,
            'percentiles': percentiles,
            'stats': stats,
            'has_image_bytes': image_bytes_col is not None
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

    # Columns to skip during serialization (binary data)
    skip_cols = set()
    if dataset_state.get('image_bytes_column'):
        skip_cols.add(dataset_state['image_bytes_column'])

    # Extract image data
    images = []
    for idx, row in page_df.iterrows():
        image_data = {'_index': int(idx)}

        # Get all columns (except binary data columns)
        for col in df.columns:
            if col in skip_cols:
                continue
            val = row[col]
            image_data[col] = _serialize_value(val)

        # Remove None entries that signal skip (binary data)
        image_data = {k: v for k, v in image_data.items() if v is not _SKIP_SENTINEL}
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

    # Columns to skip during serialization (binary data)
    skip_cols = set()
    if dataset_state.get('image_bytes_column'):
        skip_cols.add(dataset_state['image_bytes_column'])

    # Extract image data
    images = []
    for idx, row in page_df.iterrows():
        image_data = {'_index': int(idx)}

        # Get all columns (except binary data columns)
        for col in df.columns:
            if col in skip_cols:
                continue
            val = row[col]
            image_data[col] = _serialize_value(val)

        image_data = {k: v for k, v in image_data.items() if v is not _SKIP_SENTINEL}
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
    """Serve an image from the filesystem."""
    # Handle both absolute and relative paths
    if not image_path.startswith('/'):
        image_path = '/' + image_path

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404

    try:
        mimetype = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
        return send_file(image_path, mimetype=mimetype)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image-bytes/<int:index>')
def serve_image_bytes(index):
    """Serve a binary image from the dataset's image_bytes column."""
    if dataset_state['df'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    if not dataset_state['has_image_bytes']:
        return jsonify({'error': 'No binary image data in dataset'}), 400

    df = dataset_state['df']
    if index < 0 or index >= len(df):
        return jsonify({'error': 'Index out of range'}), 404

    col = dataset_state['image_bytes_column']
    image_data = df.at[index, col] if index in df.index else df.iloc[index][col]

    if image_data is None or (isinstance(image_data, float) and np.isnan(image_data)):
        return jsonify({'error': 'No image data for this index'}), 404

    # Extract raw bytes - handles both flat bytes and list<binary>/ndarray wrapping
    raw_bytes = _extract_binary_element(image_data)
    if raw_bytes is None:
        return jsonify({'error': f'Could not extract image bytes (type: {type(image_data).__name__})'}), 500

    mimetype = guess_image_mimetype(raw_bytes)
    return send_file(
        io.BytesIO(raw_bytes),
        mimetype=mimetype,
        download_name=f'image_{index}.{mimetype.split("/")[-1]}'
    )


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

    # Columns to skip during serialization (binary data)
    skip_cols = set()
    if dataset_state.get('image_bytes_column'):
        skip_cols.add(dataset_state['image_bytes_column'])

    for col in df.columns:
        if col in skip_cols:
            continue
        val = row[col]
        serialized = _serialize_value(val)
        if serialized is not _SKIP_SENTINEL:
            sample_data[col] = serialized

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
        # Exclude binary columns from JSON export
        export_df = filtered_df
        if dataset_state.get('image_bytes_column'):
            export_cols = [c for c in filtered_df.columns
                           if c != dataset_state['image_bytes_column']]
            export_df = filtered_df[export_cols]
        # Convert ndarray values (from parquet list<T> columns) to Python lists
        records = export_df.to_dict(orient='records')
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, np.ndarray):
                    rec[k] = v.tolist()
        return jsonify({
            'filtered_count': len(export_df),
            'data': records
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
