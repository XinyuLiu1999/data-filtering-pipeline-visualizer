/**
 * Data Filtering Pipeline Visualizer - Frontend Application
 */

// Global state
const state = {
    loaded: false,
    totalRecords: 0,
    numericColumns: [],
    allColumns: [],
    imageColumn: null,
    idColumn: null,
    percentiles: {},
    stats: {},
    filters: [],
    appliedFilters: [],  // Filters that have been applied via "Apply Filters" button
    filteredCount: 0,
    filteredOutCount: 0,  // Count of images filtered out
    currentPage: 1,
    currentImageView: 'retained',  // 'retained' or 'filtered-out'
    histograms: {},
    dataSource: 'huggingface'  // 'huggingface' or 'local'
};

// API base URL
const API_BASE = '';

// ============== Data Source Switching ==============

function switchDataSource(source) {
    state.dataSource = source;

    // Update tab styling
    document.querySelectorAll('.source-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.source === source);
    });

    // Show/hide source content
    document.getElementById('hf-source').style.display = source === 'huggingface' ? 'block' : 'none';
    document.getElementById('local-source').style.display = source === 'local' ? 'block' : 'none';
}

async function loadDefaultDataset() {
    try {
        showStatus('load-status', 'Fetching default dataset info...', '');

        const response = await fetch(`${API_BASE}/api/default-dataset`);
        const data = await response.json();

        if (data.repo_id) {
            document.getElementById('hf-repo').value = data.repo_id;
            document.getElementById('hf-split').value = data.split || 'train';
            document.getElementById('hf-config').value = data.config || '';

            // Set a reasonable default limit for demo
            document.getElementById('record-limit').value = '1000';

            showStatus('load-status', 'Default dataset configured. Click "Load Dataset" to load.', 'success');
        }
    } catch (error) {
        showStatus('load-status', 'Failed to fetch default dataset info', 'error');
    }
}

// ============== Data Loading ==============

async function loadDataset() {
    const limitInput = document.getElementById('record-limit').value;
    const limit = limitInput ? parseInt(limitInput) : null;

    let endpoint, body;

    if (state.dataSource === 'huggingface') {
        const repoId = document.getElementById('hf-repo').value.trim();
        const split = document.getElementById('hf-split').value.trim() || 'train';
        const config = document.getElementById('hf-config').value.trim() || null;

        if (!repoId) {
            showStatus('load-status', 'Please enter a HuggingFace dataset repository', 'error');
            return;
        }

        endpoint = `${API_BASE}/api/load-hf`;
        body = { repo_id: repoId, split, config, limit };
    } else {
        const filePath = document.getElementById('file-path').value.trim();

        if (!filePath) {
            showStatus('load-status', 'Please enter a file path', 'error');
            return;
        }

        endpoint = `${API_BASE}/api/load`;
        body = { file_path: filePath, limit };
    }

    showStatus('load-status', 'Loading dataset...', '');

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to load dataset');
        }

        // Update state
        state.loaded = true;
        state.totalRecords = data.total_records;
        state.numericColumns = data.numeric_columns;
        state.allColumns = data.all_columns;
        state.imageColumn = data.image_column;
        state.idColumn = data.id_column;
        state.percentiles = data.percentiles;
        state.stats = data.stats;
        state.filteredCount = data.total_records;

        // Update UI
        const sourceInfo = data.source === 'huggingface' ? ` from ${data.repo_id}` : '';
        showStatus('load-status', `Loaded ${data.total_records.toLocaleString()} records${sourceInfo}`, 'success');
        document.getElementById('header-stats').innerHTML =
            `<span id="total-samples">${data.total_records.toLocaleString()} samples | ${data.numeric_columns.length} metrics</span>`;

        // Show panels and update UI elements
        document.getElementById('filters-panel').style.display = 'block';
        document.getElementById('export-panel').style.display = 'block';
        document.getElementById('overview-placeholder').style.display = 'none';
        document.getElementById('overview-content').style.display = 'block';

        // Initialize UI components
        initializeFilters();
        initializeDistributionSelector();
        initializeCorrelationCheckboxes();
        initializeSortSelector();
        refreshPresets();
        updateOverview();
        updateExportStats();

    } catch (error) {
        showStatus('load-status', error.message, 'error');
    }
}

function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = 'status-message' + (type ? ` ${type}` : '');
}

// ============== Filters ==============

function initializeFilters() {
    document.getElementById('filters-container').innerHTML = '';
    state.filters = [];
}

function addFilter() {
    const filterId = Date.now();
    const filterHtml = `
        <div class="filter-rule" id="filter-${filterId}">
            <div class="filter-rule-header">
                <select id="filter-col-${filterId}" onchange="updateFilterStats(${filterId})">
                    <option value="">Select metric...</option>
                    ${state.numericColumns.map(col =>
                        `<option value="${col}">${col}</option>`
                    ).join('')}
                </select>
                <button class="filter-remove" onclick="removeFilter(${filterId})">&times;</button>
            </div>
            <div class="filter-options">
                <select id="filter-type-${filterId}">
                    <option value="absolute">Absolute</option>
                    <option value="percentile">Percentile</option>
                </select>
                <select id="filter-op-${filterId}">
                    <option value="gte">&ge; (min)</option>
                    <option value="lte">&le; (max)</option>
                    <option value="gt">&gt;</option>
                    <option value="lt">&lt;</option>
                    <option value="between">Between</option>
                </select>
                <input type="number" id="filter-val-${filterId}" placeholder="Value" step="any">
                <input type="number" id="filter-val2-${filterId}" placeholder="Max (for between)" step="any" style="display:none;">
            </div>
            <div class="filter-hint" id="filter-hint-${filterId}" style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem;"></div>
        </div>
    `;

    document.getElementById('filters-container').insertAdjacentHTML('beforeend', filterHtml);

    // Add event listener for between operator
    document.getElementById(`filter-op-${filterId}`).addEventListener('change', function() {
        const val2Input = document.getElementById(`filter-val2-${filterId}`);
        val2Input.style.display = this.value === 'between' ? 'block' : 'none';
    });

    state.filters.push({ id: filterId });
}

function removeFilter(filterId) {
    document.getElementById(`filter-${filterId}`).remove();
    state.filters = state.filters.filter(f => f.id !== filterId);
}

function updateFilterStats(filterId) {
    const column = document.getElementById(`filter-col-${filterId}`).value;
    const hintElement = document.getElementById(`filter-hint-${filterId}`);

    if (!column || !state.stats[column]) {
        hintElement.textContent = '';
        return;
    }

    const s = state.stats[column];
    const p = state.percentiles[column];
    hintElement.innerHTML = `
        Range: ${s.min.toFixed(4)} - ${s.max.toFixed(4)} |
        Mean: ${s.mean.toFixed(4)} |
        P5: ${p['5'].toFixed(4)} | P95: ${p['95'].toFixed(4)}
    `;
}

function collectFilters() {
    const filters = [];

    for (const filter of state.filters) {
        const column = document.getElementById(`filter-col-${filter.id}`)?.value;
        const type = document.getElementById(`filter-type-${filter.id}`)?.value;
        const operator = document.getElementById(`filter-op-${filter.id}`)?.value;
        const value = parseFloat(document.getElementById(`filter-val-${filter.id}`)?.value);
        const value2 = parseFloat(document.getElementById(`filter-val2-${filter.id}`)?.value);

        if (!column || isNaN(value)) continue;

        const filterObj = { column, type, operator, value };

        if (operator === 'between' && !isNaN(value2)) {
            filterObj.value = [value, value2];
        }

        filters.push(filterObj);
    }

    return filters;
}

async function applyFilters() {
    if (!state.loaded) return;

    const filters = collectFilters();

    try {
        const response = await fetch(`${API_BASE}/api/filter`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filters })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to apply filters');
        }

        // Store applied filters for use in image browser
        state.appliedFilters = [...filters];
        state.filteredCount = data.after_count;
        state.filteredOutCount = data.before_count - data.after_count;
        state.histograms = data.histograms;

        // Update UI
        updateExportStats();
        updateOverviewStats();
        updateComparisonGrid(data.before_stats, data.after_stats, data.before_count, data.after_count);
        updateDistribution();

        // Reset to first page of images
        state.currentPage = 1;
        loadCurrentImageView();

    } catch (error) {
        console.error('Filter error:', error);
        alert('Error applying filters: ' + error.message);
    }
}

function clearFilters() {
    initializeFilters();
    state.appliedFilters = [];  // Clear applied filters
    state.filteredCount = state.totalRecords;
    state.filteredOutCount = 0;
    state.histograms = {};
    updateExportStats();
    updateOverview();
    state.currentPage = 1;
    loadCurrentImageView();
}

// ============== Filter Presets ==============

async function refreshPresets() {
    try {
        const response = await fetch(`${API_BASE}/api/presets`);
        const data = await response.json();

        const select = document.getElementById('preset-select');
        select.innerHTML = '<option value="">-- Load Preset --</option>';

        for (const preset of data.presets) {
            const option = document.createElement('option');
            option.value = preset.name;
            option.textContent = `${preset.name} (${preset.filter_count} filters)`;
            option.title = preset.description || '';
            select.appendChild(option);
        }
    } catch (error) {
        console.error('Error loading presets:', error);
    }
}

async function onPresetSelect() {
    const select = document.getElementById('preset-select');
    const presetName = select.value;

    if (!presetName) return;

    try {
        const response = await fetch(`${API_BASE}/api/presets/${encodeURIComponent(presetName)}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to load preset');
        }

        // Clear existing filters and apply preset filters
        initializeFilters();

        for (const filter of data.filters) {
            addFilter();
            const filterId = state.filters[state.filters.length - 1].id;

            // Set filter values
            document.getElementById(`filter-col-${filterId}`).value = filter.column;
            document.getElementById(`filter-type-${filterId}`).value = filter.type;
            document.getElementById(`filter-op-${filterId}`).value = filter.operator;

            if (filter.operator === 'between' && Array.isArray(filter.value)) {
                document.getElementById(`filter-val-${filterId}`).value = filter.value[0];
                document.getElementById(`filter-val2-${filterId}`).value = filter.value[1];
                document.getElementById(`filter-val2-${filterId}`).style.display = 'block';
            } else {
                document.getElementById(`filter-val-${filterId}`).value = filter.value;
            }

            updateFilterStats(filterId);
        }

        // Update preset name input
        document.getElementById('preset-name').value = presetName;

    } catch (error) {
        alert('Error loading preset: ' + error.message);
    }
}

async function saveCurrentPreset() {
    const name = document.getElementById('preset-name').value.trim();

    if (!name) {
        alert('Please enter a preset name');
        return;
    }

    const filters = collectFilters();

    if (filters.length === 0) {
        alert('No filters to save');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/presets`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                filters: filters,
                description: `${filters.length} filter(s)`
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to save preset');
        }

        alert(`Preset "${data.name}" saved successfully`);
        refreshPresets();

    } catch (error) {
        alert('Error saving preset: ' + error.message);
    }
}

async function deleteSelectedPreset() {
    const select = document.getElementById('preset-select');
    const presetName = select.value;

    if (!presetName) {
        alert('Please select a preset to delete');
        return;
    }

    if (!confirm(`Delete preset "${presetName}"?`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/presets/${encodeURIComponent(presetName)}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to delete preset');
        }

        refreshPresets();
        document.getElementById('preset-name').value = '';

    } catch (error) {
        alert('Error deleting preset: ' + error.message);
    }
}

// ============== Overview ==============

function updateOverviewStats() {
    // Update just the stats cards and filter impact (without resetting comparison grid)
    const activeFilters = collectFilters().length;
    const statsHtml = `
        <div class="stat-card">
            <div class="stat-card-title">Total Samples</div>
            <div class="stat-card-value">${state.totalRecords.toLocaleString()}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Filtered Samples</div>
            <div class="stat-card-value">${state.filteredCount.toLocaleString()}</div>
            <div class="stat-card-change ${state.filteredCount < state.totalRecords ? 'negative' : 'positive'}">
                ${((state.filteredCount / state.totalRecords) * 100).toFixed(1)}% retained
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Numeric Metrics</div>
            <div class="stat-card-value">${state.numericColumns.length}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Active Filters</div>
            <div class="stat-card-value">${activeFilters}</div>
        </div>
    `;

    document.getElementById('overview-stats').innerHTML = statsHtml;

    // Update filter impact section
    updateFilterImpact(state.totalRecords, state.filteredCount);
}

function updateOverview() {
    // Full overview update (used on initial load)
    updateOverviewStats();

    // Initialize comparison grid with no filter results
    updateComparisonGrid({}, {}, state.totalRecords, state.filteredCount);
}

function updateFilterImpact(beforeCount, afterCount) {
    const retentionRate = beforeCount > 0 ? (afterCount / beforeCount) * 100 : 100;
    const removedRate = 100 - retentionRate;
    const removedCount = beforeCount - afterCount;
    const activeFilters = collectFilters().length;

    // Update impact bar
    document.getElementById('impact-before-count').textContent = beforeCount.toLocaleString();
    document.getElementById('impact-after-count').textContent = afterCount.toLocaleString();
    document.getElementById('impact-bar-after').style.width = `${retentionRate}%`;
    document.getElementById('impact-retained').textContent = `${retentionRate.toFixed(1)}% retained`;
    document.getElementById('impact-removed').textContent = `${removedRate.toFixed(1)}% removed`;

    // Update detail items
    document.getElementById('detail-retained').textContent = afterCount.toLocaleString();
    document.getElementById('detail-removed').textContent = removedCount.toLocaleString();
    document.getElementById('detail-filters').textContent = activeFilters;
}

function updateComparisonGrid(beforeStats, afterStats, beforeCount, afterCount) {
    const columns = state.numericColumns.slice(0, 12); // Limit to 12 columns
    let html = '';

    // Check if we have actual filter results or just initial state
    const hasFilterResults = Object.keys(beforeStats).length > 0;

    for (const col of columns) {
        const before = beforeStats[col] || state.stats[col] || {};
        const after = afterStats[col] || (hasFilterResults ? {} : state.stats[col]) || {};

        // Calculate changes for visual indicators
        const meanChange = before.mean && after.mean ? ((after.mean - before.mean) / before.mean * 100) : 0;
        const medianChange = before.median && after.median ? ((after.median - before.median) / before.median * 100) : 0;

        html += `
            <div class="comparison-card">
                <h4 title="${col}">${col}</h4>
                <div class="comparison-row">
                    <span class="comparison-label">Count</span>
                    <span class="comparison-before">${(before.count || beforeCount).toLocaleString()}</span>
                    <span class="comparison-after">${(after.count !== undefined ? after.count : afterCount).toLocaleString()}</span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Mean</span>
                    <span class="comparison-before">${formatNumber(before.mean)}</span>
                    <span class="comparison-after">
                        ${formatNumber(after.mean)}
                        ${hasFilterResults && meanChange !== 0 ? `<span class="comparison-change ${meanChange > 0 ? 'up' : 'down'}">${meanChange > 0 ? '+' : ''}${meanChange.toFixed(1)}%</span>` : ''}
                    </span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Std</span>
                    <span class="comparison-before">${formatNumber(before.std)}</span>
                    <span class="comparison-after">${formatNumber(after.std)}</span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Median</span>
                    <span class="comparison-before">${formatNumber(before.median)}</span>
                    <span class="comparison-after">
                        ${formatNumber(after.median)}
                        ${hasFilterResults && medianChange !== 0 ? `<span class="comparison-change ${medianChange > 0 ? 'up' : 'down'}">${medianChange > 0 ? '+' : ''}${medianChange.toFixed(1)}%</span>` : ''}
                    </span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Min</span>
                    <span class="comparison-before">${formatNumber(before.min)}</span>
                    <span class="comparison-after">${formatNumber(after.min)}</span>
                </div>
                <div class="comparison-row">
                    <span class="comparison-label">Max</span>
                    <span class="comparison-before">${formatNumber(before.max)}</span>
                    <span class="comparison-after">${formatNumber(after.max)}</span>
                </div>
            </div>
        `;
    }

    document.getElementById('comparison-grid').innerHTML = html;

    // Also update filter impact
    updateFilterImpact(beforeCount, afterCount);
}

function formatNumber(val) {
    if (val === undefined || val === null || isNaN(val)) return 'N/A';
    if (Math.abs(val) >= 1000) return val.toLocaleString(undefined, {maximumFractionDigits: 2});
    if (Math.abs(val) < 0.0001 && val !== 0) return val.toExponential(2);
    return val.toFixed(4);
}

function updateExportStats() {
    document.getElementById('export-count').textContent = state.filteredCount.toLocaleString();
    const rate = state.totalRecords > 0 ? (state.filteredCount / state.totalRecords * 100) : 0;
    document.getElementById('export-rate').textContent = rate.toFixed(1) + '%';
}

// ============== Distributions ==============

async function updateAllDistributions() {
    if (!state.loaded || state.numericColumns.length === 0) return;

    const showFiltered = document.getElementById('show-filtered').checked;
    const container = document.getElementById('distributions-grid');

    // Create cards for each metric
    let html = '';
    for (const col of state.numericColumns) {
        const stats = state.stats[col] || {};
        html += `
            <div class="distribution-card">
                <div class="distribution-card-header">
                    <span class="distribution-card-title" title="${col}">${col}</span>
                    <div class="distribution-card-stats">
                        <span>μ: ${stats.mean?.toFixed(2) || 'N/A'}</span>
                        <span>σ: ${stats.std?.toFixed(2) || 'N/A'}</span>
                    </div>
                </div>
                <div class="distribution-chart-container" id="dist-chart-${col.replace(/[^a-zA-Z0-9]/g, '_')}"></div>
            </div>
        `;
    }
    container.innerHTML = html;

    // Render each chart
    for (const col of state.numericColumns) {
        await renderDistributionChart(col, showFiltered);
    }
}

async function renderDistributionChart(column, showFiltered) {
    const chartId = `dist-chart-${column.replace(/[^a-zA-Z0-9]/g, '_')}`;
    const chartDiv = document.getElementById(chartId);
    if (!chartDiv) return;

    // Fetch histogram data
    const response = await fetch(`${API_BASE}/api/histogram/${column}?bins=30`);
    const data = await response.json();

    // Get threshold lines from current filters
    const columnFilters = state.appliedFilters.filter(f => f.column === column);

    // Create traces
    const traces = [];

    // Before histogram
    const beforeHist = state.histograms[column]?.before || data.histogram;
    traces.push({
        x: beforeHist.bins,
        y: beforeHist.counts,
        type: 'bar',
        name: 'Before',
        marker: { color: 'rgba(74, 158, 255, 0.7)' },
        hovertemplate: '%{y}<extra>Before</extra>'
    });

    // After histogram (if filtered)
    if (showFiltered && state.histograms[column]?.after) {
        const afterHist = state.histograms[column].after;
        traces.push({
            x: afterHist.bins,
            y: afterHist.counts,
            type: 'bar',
            name: 'After',
            marker: { color: 'rgba(34, 197, 94, 0.7)' },
            hovertemplate: '%{y}<extra>After</extra>'
        });
    }

    // Add threshold lines
    const shapes = [];
    for (const f of columnFilters) {
        let value = f.value;
        if (f.type === 'percentile') {
            value = state.percentiles[column]?.[String(value)] || value;
        }

        if (f.operator === 'between' && Array.isArray(value)) {
            shapes.push({
                type: 'line', x0: value[0], x1: value[0], y0: 0, y1: 1,
                yref: 'paper', line: { color: '#ef4444', width: 2, dash: 'dash' }
            });
            shapes.push({
                type: 'line', x0: value[1], x1: value[1], y0: 0, y1: 1,
                yref: 'paper', line: { color: '#ef4444', width: 2, dash: 'dash' }
            });
        } else if (typeof value === 'number') {
            shapes.push({
                type: 'line', x0: value, x1: value, y0: 0, y1: 1,
                yref: 'paper', line: { color: '#ef4444', width: 2, dash: 'dash' }
            });
        }
    }

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#8b98a5', size: 10 },
        xaxis: { gridcolor: '#2f3640', showticklabels: true },
        yaxis: { gridcolor: '#2f3640', showticklabels: true },
        barmode: 'overlay',
        shapes: shapes,
        showlegend: false,
        margin: { l: 40, r: 10, t: 10, b: 30 },
        hovermode: 'closest'
    };

    Plotly.newPlot(chartId, traces, layout, { responsive: true, displayModeBar: false });
}

// Legacy function for compatibility
function initializeDistributionSelector() {
    // Don't render distributions immediately - they'll be rendered when the tab is first shown
    // This avoids layout issues when charts are rendered in a hidden container
}

function updateDistribution() {
    updateAllDistributions();
}

// Legacy function - no longer used
function updateMetricStats(stats) {
    // This function is deprecated
}

// ============== Correlation ==============

function initializeCorrelationCheckboxes() {
    const container = document.getElementById('correlation-checkboxes');
    container.innerHTML = state.numericColumns.map(col => `
        <label class="checkbox-item">
            <input type="checkbox" value="${col}" checked>
            ${col.length > 25 ? col.substring(0, 22) + '...' : col}
        </label>
    `).join('');

    updateCorrelation();
}

async function updateCorrelation() {
    const checkboxes = document.querySelectorAll('#correlation-checkboxes input:checked');
    const columns = Array.from(checkboxes).map(cb => cb.value);

    if (columns.length < 2) {
        document.getElementById('correlation-chart').innerHTML =
            '<div class="placeholder"><p>Select at least 2 metrics</p></div>';
        return;
    }

    const params = new URLSearchParams();
    columns.forEach(col => params.append('columns', col));

    const response = await fetch(`${API_BASE}/api/correlation?${params}`);
    const data = await response.json();

    // Calculate dynamic sizing based on number of columns
    const numCols = data.columns.length;
    const cellSize = Math.max(50, Math.min(80, 600 / numCols));  // Cell size between 50-80px
    const chartSize = Math.max(500, cellSize * numCols + 200);   // Minimum 500px
    const fontSize = numCols > 10 ? 8 : (numCols > 6 ? 9 : 10);  // Smaller font for more columns

    // Truncate long column names for display
    const displayNames = data.columns.map(col =>
        col.length > 15 ? col.substring(0, 12) + '...' : col
    );

    // Create heatmap
    const trace = {
        z: data.matrix,
        x: displayNames,
        y: displayNames,
        type: 'heatmap',
        colorscale: [
            [0, '#3b82f6'],
            [0.5, '#1a1f26'],
            [1, '#ef4444']
        ],
        zmin: -1,
        zmax: 1,
        text: data.matrix.map(row => row.map(val => val.toFixed(2))),
        texttemplate: '%{text}',
        textfont: { size: fontSize, color: '#e7e9ea' },
        hovertemplate: '<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    };

    const layout = {
        title: { text: 'Correlation Matrix', font: { color: '#e7e9ea', size: 16 } },
        paper_bgcolor: '#1a1f26',
        plot_bgcolor: '#1a1f26',
        font: { color: '#8b98a5' },
        xaxis: { tickangle: 45, tickfont: { size: 11 } },
        yaxis: { autorange: 'reversed', tickfont: { size: 11 } },
        margin: { l: 120, r: 50, b: 120, t: 60 },
        width: chartSize,
        height: chartSize
    };

    Plotly.newPlot('correlation-chart', [trace], layout, { responsive: false });

    // Find high correlations
    const insights = [];
    for (let i = 0; i < data.columns.length; i++) {
        for (let j = i + 1; j < data.columns.length; j++) {
            const corr = data.matrix[i][j];
            if (Math.abs(corr) > 0.7) {
                insights.push({
                    pair: `${data.columns[i]} / ${data.columns[j]}`,
                    value: corr,
                    level: Math.abs(corr) > 0.9 ? 'high' : 'medium'
                });
            }
        }
    }

    insights.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    let insightsHtml = '<h4>Highly Correlated Pairs (|r| > 0.7)</h4>';
    if (insights.length === 0) {
        insightsHtml += '<p style="color: var(--text-secondary);">No highly correlated pairs found.</p>';
    } else {
        insightsHtml += insights.slice(0, 10).map(insight => `
            <div class="insight-item">
                <span class="insight-pair">${insight.pair}</span>
                <span class="insight-value ${insight.level}">${insight.value.toFixed(3)}</span>
            </div>
        `).join('');
    }

    document.getElementById('correlation-insights').innerHTML = insightsHtml;
}

// ============== Images ==============

function initializeSortSelector() {
    const select = document.getElementById('sort-column');
    select.innerHTML = `
        <option value="">Default Order</option>
        ${state.numericColumns.map(col =>
            `<option value="${col}">${col}</option>`
        ).join('')}
    `;
}

async function loadImages(page = 1) {
    if (!state.loaded) return;

    state.currentPage = page;
    // Use applied filters (from "Apply Filters" button) instead of current DOM values
    const filters = state.appliedFilters;
    const perPage = parseInt(document.getElementById('per-page').value);
    const sortBy = document.getElementById('sort-column').value;
    const sortOrder = document.getElementById('sort-order').value;

    document.getElementById('image-grid').innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const response = await fetch(`${API_BASE}/api/images`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filters,
                page,
                per_page: perPage,
                sort_by: sortBy,
                sort_order: sortOrder
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to load images');
        }

        renderImageGrid(data.images);
        renderPagination(data.page, data.total_pages, data.total);
        updateImageFilterStatus(data.total);

    } catch (error) {
        document.getElementById('image-grid').innerHTML =
            `<div class="placeholder"><p>Error loading images: ${error.message}</p></div>`;
    }
}

function updateImageFilterStatus(totalFiltered, totalFilteredOut = null) {
    const filterCount = state.appliedFilters.length;

    // Update counts
    state.filteredCount = totalFiltered;
    if (totalFilteredOut !== null) {
        state.filteredOutCount = totalFilteredOut;
    } else {
        state.filteredOutCount = state.totalRecords - totalFiltered;
    }

    // Update showing count
    document.getElementById('image-showing-count').textContent = totalFiltered.toLocaleString();
    document.getElementById('image-filtered-out-count').textContent = state.filteredOutCount.toLocaleString();

    // Update sub-tab badges
    document.getElementById('retained-badge').textContent = totalFiltered.toLocaleString();
    document.getElementById('filtered-out-badge').textContent = state.filteredOutCount.toLocaleString();

    // Update filter badge
    const badge = document.getElementById('filter-status-badge');
    if (filterCount === 0) {
        badge.textContent = 'No filters applied';
        badge.className = 'filter-status-badge no-filter';
    } else {
        badge.textContent = `${filterCount} filter${filterCount > 1 ? 's' : ''} applied`;
        badge.className = 'filter-status-badge filtered';
    }
}

function renderImageGrid(images, isFilteredOut = false) {
    if (images.length === 0) {
        const message = isFilteredOut
            ? 'No images have been filtered out.'
            : 'No images match the current filters.';
        document.getElementById('image-grid').innerHTML =
            `<div class="placeholder"><p>${message}</p></div>`;
        return;
    }

    const html = images.map(img => {
        // Get image path
        let imagePath = '';
        if (state.imageColumn && img[state.imageColumn]) {
            const imgVal = img[state.imageColumn];
            imagePath = Array.isArray(imgVal) ? imgVal[0] : imgVal;
        }

        // Get ID
        const id = state.idColumn ? img[state.idColumn] : img._index;

        // Get a few key metrics to display
        const metricsToShow = state.numericColumns.slice(0, 3);
        const metricBadges = metricsToShow.map(col => {
            const val = img[col];
            if (val != null && typeof val === 'number' && !isNaN(val)) {
                return `<span class="metric-badge">${col.split('_').slice(-2).join('_')}: ${val.toFixed(2)}</span>`;
            }
            return '';
        }).join('');

        const cardClass = isFilteredOut ? 'image-card filtered-out' : 'image-card';

        return `
            <div class="${cardClass}" onclick="openImageModal(${img._index})">
                ${isFilteredOut ? '<div class="filtered-out-badge">Filtered Out</div>' : ''}
                <img src="/api/image/${imagePath}" alt="${id}" loading="lazy"
                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22150%22><rect fill=%22%23242a33%22 width=%22200%22 height=%22150%22/><text fill=%22%238b98a5%22 x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22>No Image</text></svg>'">
                <div class="image-card-info">
                    <div class="image-card-id">ID: ${id}</div>
                    <div class="image-card-metrics">${metricBadges}</div>
                </div>
            </div>
        `;
    }).join('');

    document.getElementById('image-grid').innerHTML = html;
}

function renderPagination(currentPage, totalPages, totalItems) {
    const perPage = parseInt(document.getElementById('per-page').value);
    const start = (currentPage - 1) * perPage + 1;
    const end = Math.min(currentPage * perPage, totalItems);

    let html = `
        <button class="page-btn" onclick="loadImages(1)" ${currentPage === 1 ? 'disabled' : ''}>First</button>
        <button class="page-btn" onclick="loadImages(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>Prev</button>
    `;

    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);

    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="loadImages(${i})">${i}</button>`;
    }

    html += `
        <button class="page-btn" onclick="loadImages(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>Next</button>
        <button class="page-btn" onclick="loadImages(${totalPages})" ${currentPage === totalPages ? 'disabled' : ''}>Last</button>
        <span class="page-info">${start}-${end} of ${totalItems.toLocaleString()}</span>
    `;

    document.getElementById('pagination').innerHTML = html;
}

// ============== Image View Switching ==============

function switchImageView(view) {
    state.currentImageView = view;
    state.currentPage = 1;

    // Update sub-tab buttons
    document.querySelectorAll('.image-sub-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === view);
    });

    // Load appropriate images
    loadCurrentImageView();
}

function loadCurrentImageView() {
    if (state.currentImageView === 'retained') {
        loadImages(state.currentPage);
    } else {
        loadFilteredOutImages(state.currentPage);
    }
}

async function loadFilteredOutImages(page = 1) {
    if (!state.loaded) return;

    state.currentPage = page;
    const filters = state.appliedFilters;
    const perPage = parseInt(document.getElementById('per-page').value);
    const sortBy = document.getElementById('sort-column').value;
    const sortOrder = document.getElementById('sort-order').value;

    document.getElementById('image-grid').innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const response = await fetch(`${API_BASE}/api/images/filtered-out`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filters,
                page,
                per_page: perPage,
                sort_by: sortBy,
                sort_order: sortOrder
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to load filtered out images');
        }

        renderImageGrid(data.images, true);  // true indicates filtered-out view
        renderPaginationFilteredOut(data.page, data.total_pages, data.total);

    } catch (error) {
        document.getElementById('image-grid').innerHTML =
            `<div class="placeholder"><p>Error loading images: ${error.message}</p></div>`;
    }
}

function renderPaginationFilteredOut(currentPage, totalPages, totalItems) {
    const perPage = parseInt(document.getElementById('per-page').value);
    const start = (currentPage - 1) * perPage + 1;
    const end = Math.min(currentPage * perPage, totalItems);

    let html = `
        <button class="page-btn" onclick="loadFilteredOutImages(1)" ${currentPage === 1 ? 'disabled' : ''}>First</button>
        <button class="page-btn" onclick="loadFilteredOutImages(${currentPage - 1})" ${currentPage === 1 ? 'disabled' : ''}>Prev</button>
    `;

    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);

    for (let i = startPage; i <= endPage; i++) {
        html += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="loadFilteredOutImages(${i})">${i}</button>`;
    }

    html += `
        <button class="page-btn" onclick="loadFilteredOutImages(${currentPage + 1})" ${currentPage === totalPages ? 'disabled' : ''}>Next</button>
        <button class="page-btn" onclick="loadFilteredOutImages(${totalPages})" ${currentPage === totalPages ? 'disabled' : ''}>Last</button>
        <span class="page-info">${totalItems > 0 ? start : 0}-${end} of ${totalItems.toLocaleString()}</span>
    `;

    document.getElementById('pagination').innerHTML = html;
}

async function openImageModal(index) {
    try {
        const response = await fetch(`${API_BASE}/api/sample/${index}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to load sample');
        }

        // Set image
        let imagePath = '';
        if (state.imageColumn && data[state.imageColumn]) {
            const imgVal = data[state.imageColumn];
            imagePath = Array.isArray(imgVal) ? imgVal[0] : imgVal;
        }
        document.getElementById('modal-image').src = `/api/image/${imagePath}`;

        // Metadata (non-numeric columns)
        const metadataHtml = state.allColumns
            .filter(col => !state.numericColumns.includes(col) && col !== '_index' && col !== '_percentile_ranks')
            .map(col => {
                let val = data[col];
                if (val === null || val === undefined) val = 'N/A';
                if (typeof val === 'object') val = JSON.stringify(val);
                if (typeof val === 'string' && val.length > 100) val = val.substring(0, 100) + '...';
                return `
                    <div class="detail-row">
                        <span class="detail-key">${col}</span>
                        <span class="detail-value">${val}</span>
                    </div>
                `;
            }).join('');
        document.getElementById('modal-metadata').innerHTML = metadataHtml || '<p>No metadata</p>';

        // Metrics (numeric columns)
        const metricsHtml = state.numericColumns.map(col => {
            const val = data[col];
            const displayVal = (val != null && typeof val === 'number' && !isNaN(val)) ? val.toFixed(6) : 'N/A';
            return `
                <div class="detail-row">
                    <span class="detail-key">${col}</span>
                    <span class="detail-value">${displayVal}</span>
                </div>
            `;
        }).join('');
        document.getElementById('modal-metrics').innerHTML = metricsHtml;

        // Percentile ranks
        const ranks = data._percentile_ranks || {};
        const percentilesHtml = state.numericColumns.map(col => {
            const rank = ranks[col];
            if (rank == null || typeof rank !== 'number' || isNaN(rank)) return '';
            return `
                <div class="percentile-bar">
                    <span class="percentile-name" title="${col}">${col}</span>
                    <div class="percentile-track">
                        <div class="percentile-fill" style="width: ${rank}%"></div>
                    </div>
                    <span class="percentile-value">${rank.toFixed(1)}%</span>
                </div>
            `;
        }).join('');
        document.getElementById('modal-percentiles').innerHTML = percentilesHtml;

        // Show modal
        document.getElementById('image-modal').classList.add('active');

    } catch (error) {
        alert('Error loading sample: ' + error.message);
    }
}

function closeModal() {
    document.getElementById('image-modal').classList.remove('active');
}

// ============== Export ==============

async function exportIndices() {
    const filters = collectFilters();

    try {
        const response = await fetch(`${API_BASE}/api/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filters, format: 'indices' })
        });

        const data = await response.json();

        // Create download
        const blob = new Blob([JSON.stringify(data.ids, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'filtered_ids.json';
        a.click();
        URL.revokeObjectURL(url);

    } catch (error) {
        alert('Export error: ' + error.message);
    }
}

async function exportJSON() {
    const filters = collectFilters();

    try {
        const response = await fetch(`${API_BASE}/api/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filters, format: 'json' })
        });

        const data = await response.json();

        // Create download
        const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'filtered_data.json';
        a.click();
        URL.revokeObjectURL(url);

    } catch (error) {
        alert('Export error: ' + error.message);
    }
}

// ============== Tab Navigation ==============

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Update tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.toggle('active', pane.id === `${tabName}-tab`);
    });

    // Load data for specific tabs
    if (tabName === 'images' && state.loaded) {
        loadCurrentImageView();
    }

    // Re-render distribution charts after tab becomes visible
    if (tabName === 'distributions' && state.loaded) {
        // Use setTimeout to allow the DOM to update, then re-render charts
        setTimeout(() => {
            updateAllDistributions();
        }, 50);
    }
}

// ============== Keyboard Shortcuts ==============

document.addEventListener('keydown', function(e) {
    // Close modal on Escape
    if (e.key === 'Escape') {
        closeModal();
    }
});

// Close modal on outside click
document.getElementById('image-modal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeModal();
    }
});

// ============== Initialize ==============

// Set default file path from example
document.getElementById('file-path').value = '/cephfs/liuxinyu/cc3m-parquet/cc3m-first50-merged.jsonl';
