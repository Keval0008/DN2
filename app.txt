import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
from io import BytesIO
import base64
from dash.dependencies import ALL
from sklearn.metrics import jaccard_score
import numpy as np
import uuid

# Function to calculate Jaccard similarity between two sets
def jaccard_similarity(set1, set2):
    set1, set2 = set(set1), set(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Match categorical columns between original and transformed datasets
def match_categorical_columns(original_df, transformed_df, threshold=0.9):
    matches = []
    for orig_col in original_df.select_dtypes(include=['object', 'category']).columns:
        match_info = {'original_column': orig_col, 'transformed_column': None, 'status': 'no_match', 'similarity': 0}
        
        # Check for exact name match first
        if orig_col in transformed_df.columns:
            similarity = jaccard_similarity(original_df[orig_col].dropna(), transformed_df[orig_col].dropna())
            if similarity >= threshold:
                match_info['transformed_column'] = orig_col
                match_info['status'] = 'single_match'
                match_info['similarity'] = similarity
                matches.append(match_info)
                continue
        
        # Fuzzy matching with other columns
        potential_matches = []
        for trans_col in transformed_df.select_dtypes(include=['object', 'category']).columns:
            similarity = jaccard_similarity(original_df[orig_col].dropna(), transformed_df[trans_col].dropna())
            if similarity >= threshold:
                potential_matches.append((trans_col, similarity))
        
        if len(potential_matches) == 1:
            match_info['transformed_column'] = potential_matches[0][0]
            match_info['status'] = 'single_match'
            match_info['similarity'] = potential_matches[0][1]
        elif len(potential_matches) > 1:
            match_info['status'] = 'multiple_matches'
            match_info['potential_matches'] = potential_matches
        matches.append(match_info)
    
    return matches

# Group and aggregate numerical columns
def group_and_aggregate(original_df, transformed_df, cat_matches, num_columns):
    matched_cols = [(m['original_column'], m['transformed_column']) for m in cat_matches 
                    if m['status'] == 'single_match' and m['transformed_column'] is not None]
    if not matched_cols:
        return None, None
    
    orig_group_cols, trans_group_cols = zip(*matched_cols)
    orig_agg = original_df.groupby(list(orig_group_cols))[num_columns].sum().reset_index()
    trans_agg = transformed_df.groupby(list(trans_group_cols))[num_columns].sum().reset_index()
    return orig_agg, trans_agg

# Compare aggregated results and flag anomalies
def compare_and_flag_anomalies(orig_agg, trans_agg, num_columns, threshold):
    if orig_agg is None or trans_agg is None:
        return None
    
    # Merge on group columns
    group_cols = [col for col in orig_agg.columns if col not in num_columns]
    merged = orig_agg.merge(trans_agg, on=group_cols, suffixes=('_orig', '_trans'))
    
    results = []
    for num_col in num_columns:
        diff_col = f'{num_col}_diff'
        merged[diff_col] = abs(merged[f'{num_col}_orig'] - merged[f'{num_col}_trans'])
        merged[f'{num_col}_anomaly'] = merged[diff_col] > threshold
        results.append(merged[group_cols + [f'{num_col}_orig', f'{num_col}_trans', diff_col, f'{num_col}_anomaly']])
    
    return pd.concat(results, axis=1).loc[:, ~pd.DataFrame().columns.duplicated()]

# Convert DataFrame to Excel
def df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return base64.b64encode(output.getvalue()).decode()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'])

app.layout = html.Div(className='container mx-auto p-4 max-w-6xl', children=[
    html.H1('Data Reconciliation Tool', className='text-2xl font-bold mb-4 text-center'),
    
    # File upload section
    html.Div(className='mb-6', children=[
        html.H2('Upload Datasets', className='text-lg font-semibold mb-2'),
        dcc.Upload(
            id='upload-original',
            children=html.Button('Upload Original Dataset (CSV)', className='bg-blue-500 text-white px-4 py-2 rounded mr-2'),
            accept='.csv',
        ),
        dcc.Upload(
            id='upload-transformed',
            children=html.Button('Upload Transformed Dataset (CSV)', className='bg-blue-500 text-white px-4 py-2 rounded'),
            accept='.csv',
        ),
    ]),
    
    # Categorical column selection
    html.Div(id='cat-selection-container', className='mb-6', children=[
        html.H2('Select Categorical Columns', className='text-lg font-semibold mb-2'),
        html.Label('Original Dataset Columns:', className='block mb-1'),
        dcc.Dropdown(id='cat-columns', multi=True, className='mb-4'),
        html.Div(id='match-results', className='mb-4'),
    ]),
    
    # Numerical column selection and threshold
    html.Div(className='mb-6', children=[
        html.H2('Select Numerical Columns and Threshold', className='text-lg font-semibold mb-2'),
        html.Label('Numerical Columns:', className='block mb-1'),
        dcc.Dropdown(id='num-columns', multi=True, className='mb-2'),
        html.Label('Anomaly Threshold:', className='block mb-1'),
        dcc.Input(id='threshold', type='number', value=10, className='border p-2 rounded w-full mb-2'),
        html.Button('Run Reconciliation', id='run-reconciliation', className='bg-green-500 text-white px-4 py-2 rounded'),
    ]),
    
    # Results display
    html.Div(id='results-container', className='mb-6', children=[
        html.H2('Reconciliation Results', className='text-lg font-semibold mb-2'),
        dash_table.DataTable(id='results-table', style_table={'overflowX': 'auto'}),
        html.A('Download Results (Excel)', id='download-link', download='reconciliation_results.xlsx', 
               className='bg-blue-500 text-white px-4 py-2 rounded inline-block mt-2'),
    ]),
    
    # Store data
    dcc.Store(id='original-data'),
    dcc.Store(id='transformed-data'),
    dcc.Store(id='match-data'),
])

# Parse uploaded CSV
def parse_csv(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(BytesIO(decoded))
    return df

# Callbacks
@app.callback(
    [Output('cat-columns', 'options'),
     Output('original-data', 'data')],
    Input('upload-original', 'contents'),
    State('upload-original', 'filename')
)
def update_original_upload(contents, filename):
    if contents is None:
        return [], None
    df = parse_csv(contents, filename)
    cat_cols = [{'label': col, 'value': col} for col in df.select_dtypes(include=['object', 'category']).columns]
    return cat_cols, df.to_json()

@app.callback(
    Output('transformed-data', 'data'),
    Input('upload-transformed', 'contents'),
    State('upload-transformed', 'filename')
)
def update_transformed_upload(contents, filename):
    if contents is None:
        return None
    df = parse_csv(contents, filename)
    return df.to_json()

@app.callback(
    Output('match-results', 'children'),
    Output('match-data', 'data'),
    Input('cat-columns', 'value'),
    State('original-data', 'data'),
    State('transformed-data', 'data')
)
def update_matches(selected_cols, orig_json, trans_json):
    if not selected_cols or not orig_json or not trans_json:
        return "Please upload both datasets and select categorical columns.", None
    
    orig_df = pd.read_json(orig_json)
    trans_df = pd.read_json(trans_json)
    matches = match_categorical_columns(orig_df, trans_df)
    
    match_elements = []
    match_data = []
    for match in matches:
        if match['original_column'] not in selected_cols:
            continue
        if match['status'] == 'single_match':
            match_elements.append(html.P(f"{match['original_column']} → {match['transformed_column']} (Similarity: {match['similarity']:.2%})"))
            match_data.append(match)
        elif match['status'] == 'multiple_matches':
            match_elements.append(html.Div([
                html.P(f"{match['original_column']} has multiple matches:"),
                dcc.Dropdown(
                    id={'type': 'match-selector', 'index': match['original_column']},
                    options=[{'label': f"{col} (Similarity: {sim:.2%})", 'value': col} for col, sim in match['potential_matches']],
                    placeholder="Select a match",
                )
            ]))
            match_data.append(match)
        else:
            match_elements.append(html.P(f"{match['original_column']} → No match found"))
            match_data.append(match)
    
    return match_elements, match_data

@app.callback(
    Output('num-columns', 'options'),
    Input('original-data', 'data')
)
def update_num_columns(orig_json):
    if not orig_json:
        return []
    df = pd.read_json(orig_json)
    num_cols = [{'label': col, 'value': col} for col in df.select_dtypes(include=['int64', 'float64']).columns]
    return num_cols

@app.callback(
    [Output('results-table', 'data'),
     Output('results-table', 'columns'),
     Output('download-link', 'href')],
    Input('run-reconciliation', 'n_clicks'),
    Input({'type': 'match-selector', 'index': ALL}, 'value'),
    State('cat-columns', 'value'),
    State('num-columns', 'value'),
    State('threshold', 'value'),
    State('original-data', 'data'),
    State('transformed-data', 'data'),
    State('match-data', 'data')
)
def run_reconciliation(n_clicks, match_selections, cat_columns, num_columns, threshold, orig_json, trans_json, match_data):
    if not n_clicks or not orig_json or not trans_json or not cat_columns or not num_columns:
        return [], [], ""
    
    orig_df = pd.read_json(orig_json)
    trans_df = pd.read_json(trans_json)
    
    # Update matches with user selections
    updated_matches = match_data.copy()
    for match, selection in zip(updated_matches, match_selections):
        if match['status'] == 'multiple_matches' and selection:
            match['transformed_column'] = selection
            match['status'] = 'single_match'
    
    orig_agg, trans_agg = group_and_aggregate(orig_df, trans_df, updated_matches, num_columns)
    if orig_agg is None:
        return [], [], ""
    
    result_df = compare_and_flag_anomalies(orig_agg, trans_agg, num_columns, threshold)
    if result_df is None:
        return [], [], ""
    
    columns = [{'name': col, 'id': col} for col in result_df.columns]
    data = result_df.to_dict('records')
    excel_data = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{df_to_excel(result_df)}"
    
    return data, columns, excel_data

if __name__ == '__main__':
    app.run_server(debug=True)
