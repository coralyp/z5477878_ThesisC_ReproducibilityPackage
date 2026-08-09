# Import libraries
import math, random
from dataclasses import dataclass
from pathlib import Path
from neo4j import GraphDatabase

import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv


# Settings
CSV = next(Path(__file__).resolve().parents[2].rglob('crossdomain_dataset.csv'))
OUTDIR = 'ext_results_cross'
EPOCHS, LR, HIDDEN, BATCH_SIZE, SEED = 150, .01, 8, 32, 42
DEVICE = 'cpu'
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE = 'bolt://127.0.0.1:7687', 'neo4j', "Pigz4Lyfe", 'neo4j'
TEST_FRACTION = VALIDATION_FRACTION = .10
SPLIT_MODE, GROUP_COLUMN = 'ordered', 'source'
UNKNOWN_CATEGORY = '__UNKNOWN__'


# Define information for KG
INPUT_COLUMNS = ['material_type', 'material_form', 'material_diameter', 'laser_power', 'scanning_speed', 'hatching_distance', 'layer_thickness', 'rotation_angle']
INTERMEDIATE_COLUMNS = ['energy_input', 'melt_pool_behaviour', 'thermal_gradient', 'porosity_defects', 'microstructure', 'anisotropy']
TARGET_COLUMNS = ['hardness', 'yield_stress', 'tensile_strength', 'elongation_to_failure']
CATEGORICAL_COLUMNS = ['material_type', 'material_form']
NUMERICAL_INPUT_COLUMNS = [c for c in INPUT_COLUMNS if c not in CATEGORICAL_COLUMNS]
DATA_COLUMNS = INPUT_COLUMNS + TARGET_COLUMNS
NODE_ORDER = INPUT_COLUMNS + INTERMEDIATE_COLUMNS + TARGET_COLUMNS
NODE_INDEX = {n: i for i, n in enumerate(NODE_ORDER)}
NUM_NODES = len(NODE_ORDER)
NODE_FEATURE_DIM = NUM_NODES + 1
OBSERVED_VALUE_FEATURE_INDEX = NUM_NODES
TARGET_INDICES = [NODE_INDEX[c] for c in TARGET_COLUMNS]

EDGE_TABLE = EDGE_INDEX = EDGE_ATTR = None


# Set random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Setup, load, and initialise KG from Neo4j
def edge_table_to_edge_index(df):
    return torch.tensor(df[['source_index', 'target_index']].to_numpy().T, dtype=torch.long).contiguous()

def edge_table_to_edge_attr(df):
    return torch.tensor(df[['weight']].to_numpy(np.float32), dtype=torch.float32).contiguous()

def load_edge_table_from_neo4j(uri, user, password, database):
    node_q = '''MATCH (n:LPBF_Node) WHERE n.code_name IN $nodes RETURN n.code_name AS code_name ORDER BY code_name'''
    edge_q = '''MATCH (s:LPBF_Node)-[r]->(t:LPBF_Node)
                WHERE s.code_name IN $nodes AND t.code_name IN $nodes
                RETURN s.code_name AS source, coalesce(r.relationship_name, type(r)) AS relationship,
                       t.code_name AS target, coalesce(r.weight, 1.0) AS weight
                ORDER BY coalesce(r.origin, ''), s.code_name, coalesce(r.relationship_name, type(r)), t.code_name'''
    
    with GraphDatabase.driver(uri, auth=(user, password)) as driver, driver.session(database=database) as session:
        nodes = pd.DataFrame(dict(r) for r in session.run(node_q, nodes=NODE_ORDER))
        edges = pd.DataFrame(dict(r) for r in session.run(edge_q, nodes=NODE_ORDER))
        
    edges = edges.drop_duplicates(['source', 'relationship', 'target']).reset_index(drop=True)
    edges['source_index'] = edges.source.map(NODE_INDEX).astype(int)
    edges['target_index'] = edges.target.map(NODE_INDEX).astype(int)
    edges['weight'] = edges.weight.astype(float)
    return edges, nodes.drop_duplicates('code_name').reset_index(drop=True)

def initialise_knowledge_graph():
    edges, nodes = load_edge_table_from_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    edge_index, edge_attr = edge_table_to_edge_index(edges), edge_table_to_edge_attr(edges)
    info = {
        'num_nodes': len(nodes), 'num_edges': len(edges), 'edge_index_shape': list(edge_index.shape),
        'edge_weight_min': float(edges.weight.min()), 'edge_weight_max': float(edges.weight.max()),
        'edge_weight_mean': float(edges.weight.mean())
    }
    return edges, edge_index, edge_attr, nodes, info


# Creates container for scaling info for each mechanical property
@dataclass
class NaNTargetScaler:
    columns: list
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, df, columns):
        x = df[columns].to_numpy(float)
        mean, scale = np.nanmean(x, 0), np.nanstd(x, 0)
        return cls(list(columns), mean, np.where(scale > 1e-12, scale, 1.))

    def transform(self, x):
        return (np.asarray(x, float) - self.mean_) / self.scale_
    
    def inverse_transform(self, x):
        return np.asarray(x, float) * self.scale_ + self.mean_


# Creates container for all fitted preprocessing information
@dataclass
class Preprocessors:
    label_encoders: dict
    input_scaler: StandardScaler
    target_scaler: NaNTargetScaler
    target_columns: list


# Clean dataset and encode categorical inputs
def clean_numeric_columns(df):
    df = df.copy()
    for c in NUMERICAL_INPUT_COLUMNS + TARGET_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].replace(r'^\s*$', np.nan, regex=True)
    return df

def _fit_label_encoder_with_unknown(values):
    e = LabelEncoder()
    e.classes_ = np.asarray(sorted(set(values.astype(str)) | {UNKNOWN_CATEGORY}), dtype=object)
    return e

def _encode_categories(df, encoders):
    df = df.copy()
    for c, e in encoders.items():
        v = df[c].astype(str)
        df[c] = e.transform(v.where(v.isin(set(e.classes_)), UNKNOWN_CATEGORY))
    return df


# Fit preprocessing information (encoders, standardisations)
def fit_preprocessors(df, target_columns=None):
    targets = list(target_columns or TARGET_COLUMNS)
    df = df.copy()
    
    encoders = {c: _fit_label_encoder_with_unknown(df[c]) for c in CATEGORICAL_COLUMNS}
    df = _encode_categories(df, encoders)

    input_scaler = StandardScaler()
    df[INPUT_COLUMNS] = input_scaler.fit_transform(df[INPUT_COLUMNS].astype(float))

    target_scaler = NaNTargetScaler.fit(df, targets)
    df[targets] = target_scaler.transform(df[targets].to_numpy(float))

    return df, Preprocessors(encoders, input_scaler, target_scaler, targets)

def apply_preprocessors(df, pp):
    df = _encode_categories(df.copy(), pp.label_encoders)
    df[INPUT_COLUMNS] = pp.input_scaler.transform(df[INPUT_COLUMNS].astype(float))
    df[pp.target_columns] = pp.target_scaler.transform(df[pp.target_columns].to_numpy(float))
    return df


# Convert data entries to KG topology
def row_to_graph(row, target_columns=None):
    targets = list(target_columns or TARGET_COLUMNS)
    x = torch.zeros((NUM_NODES, NODE_FEATURE_DIM))
    x[:, :NUM_NODES] = torch.eye(NUM_NODES) # Creates identity matrix for node features

    # Assign data entry values to nodes
    for n in INPUT_COLUMNS: 
        x[NODE_INDEX[n], OBSERVED_VALUE_FEATURE_INDEX] = float(row[n])

    # Create target mask; assign True if target value is known, False if not
    vals = np.asarray([row[t] for t in targets], np.float32)
    mask = np.isfinite(vals)
    node_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    node_mask[[NODE_INDEX[t] for t in targets]] = True

    # Create PyTorch graph
    return Data(x=x, edge_index=EDGE_INDEX.clone(), edge_attr=EDGE_ATTR.clone(),
                y=torch.from_numpy(np.nan_to_num(vals)).unsqueeze(0), y_mask=torch.from_numpy(mask).unsqueeze(0),
                target_node_mask=node_mask)


# Create graph for each data entry
def build_dataset(df, target_columns=None):
    return [row_to_graph(row, target_columns) for _, row in df.iterrows()]


# GAT Architecture
class GATRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_target_indices):
        super().__init__()
        self.output_target_indices = list(output_target_indices)
        self.gats = nn.ModuleList([
            GATConv(in_dim, hidden_dim, heads=4, concat=True, dropout=0, edge_dim=1, fill_value=1.), # Initialise GAT Layer 1
            GATConv(hidden_dim * 4, hidden_dim, heads=2, concat=True, dropout=0, edge_dim=1, fill_value=1.), # Initialise GAT Layer 2
            GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=True, dropout=0, edge_dim=1, fill_value=1.) # Initialise GAT Layer 3
        ])
        self.fc = nn.Linear(hidden_dim * len(self.output_target_indices), len(self.output_target_indices)) # Define fully connected layer

    # Forward pass
    def forward(self, x, edge_index, edge_attr, batch):
        # Run each GAT layer + ELU activation
        for gat in self.gats:
            x = F.elu(gat(x, edge_index, edge_attr=edge_attr))
        # Keep ONLY target nodes and concatenate them
        x = x.view(-1, NUM_NODES, x.size(-1))[:, self.output_target_indices].flatten(1)
        # Fully connected layer
        return self.fc(x)


# Calculate MSE loss only if target value exists in dataset
def masked_mse_loss(pred, target, mask):
    m = mask.to(pred.dtype)
    counts = m.sum(0)
    valid = counts > 0
    if not valid.any(): 
        return pred.sum() * 0
    losses = ((pred - target).square() * m).sum(0) / counts.clamp_min(1)
    return losses[valid].mean()


# Train for one epoch: pass through GAT -> generate predictions -> compare predictions w/ target values -> calculate MSE -> backpropagate gradients -> adjust model parameters -> repeat per batch
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = masked_mse_loss(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch), batch.y, batch.y_mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


# Prediction function; no more training
@torch.no_grad()
def predict(model, loader, device, output_dim=None):
    model.eval()
    p, y, m = [], [], []

    for batch in loader:
        batch = batch.to(device)
        p.append(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).cpu().numpy())
        y.append(batch.y.cpu().numpy())
        m.append(batch.y_mask.cpu().numpy().astype(bool))

    return np.vstack(p), np.vstack(y), np.vstack(m)


# Calculate performance metrics
def inverse_targets(arr, target_scaler): 
    return target_scaler.inverse_transform(arr)

def compute_metrics_1d(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    n = len(y_true)
    if not n: 
        return dict(observed_samples=0, MAE=np.nan, MSE=np.nan, RMSE=np.nan, R2=np.nan)
    mse = mean_squared_error(y_true, y_pred)
    return dict(observed_samples=n, MAE=float(mean_absolute_error(y_true, y_pred)), MSE=float(mse),
                RMSE=float(math.sqrt(mse)), R2=float(r2_score(y_true, y_pred)) if n >= 2 else np.nan)

def compute_masked_overall_metrics(y_true, y_pred, mask):
    observed = mask.astype(bool)
    out = compute_metrics_1d(y_true[observed], y_pred[observed]) if observed.any() else compute_metrics_1d([], [])
    return out

def per_target_metrics(y_true, y_pred, mask, targets):
    return {name: compute_metrics_1d(y_true[m, i], y_pred[m, i]) for i, name in enumerate(targets) for m in [mask[:, i].astype(bool)]}

def compute_prediction_accuracy_by_target(y_true, y_pred, mask, targets, epsilon=1e-8):
    rows = []
    for i, name in enumerate(targets):
        obs = mask[:, i].astype(bool)
        actual, pred = y_true[obs, i].astype(float), y_pred[obs, i].astype(float)
        if not len(actual):
            rows.append(dict(property=name, test_samples=0, valid_percentage_error_samples=0, mean_actual=np.nan, mean_predicted=np.nan,
                             MAE=np.nan, RMSE=np.nan, R2=np.nan, MAPE_percent=np.nan, median_absolute_percentage_error_percent=np.nan,
                             mean_prediction_accuracy_percent=np.nan, median_prediction_accuracy_percent=np.nan,
                             within_5_percent_error=np.nan, within_10_percent_error=np.nan, within_20_percent_error=np.nan))
            continue
        ae = np.abs(actual - pred)
        valid = np.abs(actual) > epsilon
        ape = np.full_like(actual, np.nan)
        ape[valid] = ae[valid] / np.abs(actual[valid]) * 100
        acc = np.clip(100 - ape, 0, 100)
        mse = mean_squared_error(actual, pred)
        f = lambda fn: float(fn(ape)) if valid.any() else np.nan
        rows.append(dict(property=name, test_samples=len(actual), valid_percentage_error_samples=int(valid.sum()), mean_actual=float(actual.mean()),
                         mean_predicted=float(pred.mean()), MAE=float(mean_absolute_error(actual, pred)), RMSE=float(math.sqrt(mse)),
                         R2=float(r2_score(actual, pred)) if len(actual) >= 2 else np.nan, MAPE_percent=f(np.nanmean),
                         median_absolute_percentage_error_percent=f(np.nanmedian), mean_prediction_accuracy_percent=float(np.nanmean(acc)) if valid.any() else np.nan,
                         median_prediction_accuracy_percent=float(np.nanmedian(acc)) if valid.any() else np.nan,
                         within_5_percent_error=float(np.nanmean(ape <= 5) * 100) if valid.any() else np.nan,
                         within_10_percent_error=float(np.nanmean(ape <= 10) * 100) if valid.any() else np.nan,
                         within_20_percent_error=float(np.nanmean(ape <= 20) * 100) if valid.any() else np.nan))
    return pd.DataFrame(rows)

def add_sample_accuracy_columns(df, y_true, y_pred, mask, targets, epsilon=1e-8):
    out = df.copy()
    for i, name in enumerate(targets):
        obs, actual, pred = mask[:, i].astype(bool), y_true[:, i].astype(float), y_pred[:, i].astype(float)
        ae = np.full(len(actual), np.nan)
        pe = np.full(len(actual), np.nan)
        acc = np.full(len(actual), np.nan)
        ae[obs] = np.abs(actual[obs] - pred[obs])
        valid = obs & (np.abs(actual) > epsilon)
        pe[valid] = ae[valid] / np.abs(actual[valid]) * 100
        acc[valid] = np.clip(100 - pe[valid], 0, 100)
        out[f'observed_{name}'], out[f'absolute_error_{name}'] = obs, ae
        out[f'absolute_percentage_error_{name}'], out[f'prediction_accuracy_percent_{name}'] = pe, acc
    return out

def evaluate_scaled_mse(model, loader, device, output_dim):
    pred, y, mask = predict(model, loader, device)
    mses = [mean_squared_error(y[m, i], pred[m, i]) for i in range(output_dim) for m in [mask[:, i].astype(bool)] if m.any()]
    return float(np.mean(mses)) if mses else float('inf')


# Select optimal epoch with lowest MSE between validation and training
def select_epoch_with_validation(train_ds, validation_ds, targets, device, epochs, lr, batch_size, hidden_dim, seed):
    set_seed(seed)
    train_loader, val_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(validation_ds, batch_size=batch_size)
    model = GATRegressor(NODE_FEATURE_DIM, hidden_dim, [NODE_INDEX[t] for t in targets]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_epoch, best_val = 1, float('inf')
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_scaled_mse(model, val_loader, device, len(targets))
        if val_loss < best_val:
            best_epoch, best_val = epoch, val_loss
    return best_epoch, best_val


# Retrain final model with optimal number of epochs
def train_and_evaluate_gat(development_ds, test_ds, test_raw, pp, targets, device, model_dir, model_name, selected_epochs, lr, batch_size, hidden_dim, seed, best_validation_mse):
    set_seed(seed)
    model_dir.mkdir(parents=True, exist_ok=True)

    dev_loader, test_loader = DataLoader(development_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size)

    model = GATRegressor(NODE_FEATURE_DIM, hidden_dim, [NODE_INDEX[t] for t in targets]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(selected_epochs): 
        train_one_epoch(model, dev_loader, optimizer, device)

    pred_s, y_s, mask = predict(model, test_loader, device)
    pred, y = pp.target_scaler.inverse_transform(pred_s), pp.target_scaler.inverse_transform(y_s)
    y = np.where(mask, y, np.nan)
    
    overall = compute_masked_overall_metrics(y, pred, mask)
    by_target = per_target_metrics(y, pred, mask, targets)

    pd.DataFrame([{'property': n, **m} for n, m in by_target.items()]).to_csv(model_dir / 'per_target_metrics.csv', index=False)
    acc_df = compute_prediction_accuracy_by_target(y, pred, mask, targets)
    acc_df.to_csv(model_dir / 'per_target_prediction_accuracy.csv', index=False)

    results = dict(model=model_name, target_columns=targets, overall_metrics=overall, per_target_metrics=by_target,
                   per_target_prediction_accuracy=acc_df.to_dict('records'), selected_epoch=int(selected_epochs),
                   best_validation_mse_scaled=float(best_validation_mse), maximum_candidate_epochs=EPOCHS,
                   lr=lr, hidden_dim=hidden_dim, gat_layers=3, test_evaluations_during_training=0)
    summary = {
        'model': model_name,
        'selected_epoch': selected_epochs,
        'best_validation_mse_scaled': best_validation_mse,
        **{f'overall_{k}': v for k, v in overall.items()
           if k not in {'observed_samples', 'metric_note'} and np.isscalar(v)}
    }
    pd.DataFrame([summary]).to_csv(model_dir / 'model_summary.csv', index=False)

    pred_df = test_raw.reset_index(drop=True).copy()
    for i, t in enumerate(targets):
        pred_df[f'observed_{t}'], pred_df[f'actual_{t}'], pred_df[f'pred_{t}'] = mask[:, i], y[:, i], pred[:, i]
        pred_df[f'residual_{t}'] = np.where(mask[:, i], y[:, i] - pred[:, i], np.nan)

    return results, model


# Prediction helper for sensitivity analysis; takes raw perturbed data -> applies original preprocessors -> converts rows into graphs -> GAT prediction -> inverse target scaling -> original unit predictions
def predict_from_raw_dataframe(model, raw_df, pp, device, batch_size):
    ds = build_dataset(apply_preprocessors(raw_df.copy(), pp), pp.target_columns)
    pred, _, _ = predict(model, DataLoader(ds, batch_size=batch_size), device)
    return pp.target_scaler.inverse_transform(pred)


# Numerical sensitivity analysis
def run_numerical_sensitivity_analysis(model, train_raw, test_raw, pp, device, batch_size, outdir):
    d = outdir / 'sensitivity_analysis'
    d.mkdir(parents=True, exist_ok=True)
    baseline = predict_from_raw_dataframe(model, test_raw, pp, device, batch_size) # Get baseline == 0% change

    rows = []
    materials = test_raw.material_type.astype(str).to_numpy()

    # Perturb numerical inputs
    for inp in NUMERICAL_INPUT_COLUMNS:
        # Set boundaries based on known MIN/MAX data 
        train_vals = pd.to_numeric(train_raw[inp], errors='coerce')
        x_min, x_max = float(train_vals.min()), float(train_vals.max())
        base_x = pd.to_numeric(test_raw[inp], errors='coerce').to_numpy(float)

        # Apply perturbations [-50%, -40%, -30%, -20%, -10%, 0%, 10%, 20%, 30%, 40%, 50%]
        for pct in range(-50, 51, 10):
            requested_x = base_x * (1 + pct / 100)
            perturbed_x = np.clip(requested_x, x_min, x_max) # Clip values to boundaries
            changed = test_raw.copy()
            changed[inp] = perturbed_x
            pred = baseline if pct == 0 else predict_from_raw_dataframe(model, changed, pp, device, batch_size)
            dx = perturbed_x - base_x # Get change in input
            dx_pct = np.divide(dx, np.abs(base_x), out=np.full_like(base_x, np.nan), where=np.abs(base_x) > 1e-12) * 100 # Get change in input as %

            for j, target in enumerate(TARGET_COLUMNS):
                b, p = baseline[:, j], pred[:, j]
                dy = p - b # Get change in output
                dy_pct = np.divide(dy, np.abs(b), out=np.full_like(b, np.nan), where=np.abs(b) > 1e-8) * 100 # Get change in output %
                rows += [dict(sample_index=i, material_type=materials[i], input=inp, output=target,
                              requested_input_change_percent=pct, baseline_input=base_x[i],
                              requested_perturbed_input=requested_x[i], perturbed_input_after_clipping=perturbed_x[i],
                              actual_input_change=dx[i], actual_input_change_percent=dx_pct[i],
                              baseline_prediction=b[i], perturbed_prediction=p[i],
                              prediction_change=dy[i], prediction_change_percent=dy_pct[i])
                         for i in range(len(test_raw))]

    # Sort numerical sensitivity analysis by material
    samples = pd.DataFrame(rows).sort_values(
        ['material_type', 'output', 'input', 'requested_input_change_percent', 'sample_index']
    ).reset_index(drop=True)

    keys = ['material_type', 'input', 'output', 'requested_input_change_percent']
    summary = samples.groupby(keys, as_index=False).agg(
        mean_actual_input_change_percent=('actual_input_change_percent', 'mean'),
        mean_baseline_input=('baseline_input', 'mean'),
        mean_perturbed_input=('perturbed_input_after_clipping', 'mean'),
        mean_baseline_prediction=('baseline_prediction', 'mean'),
        mean_perturbed_prediction=('perturbed_prediction', 'mean'),
        mean_prediction_change=('prediction_change', 'mean'),
        median_prediction_change=('prediction_change', 'median'),
        std_prediction_change=('prediction_change', 'std'),
        mean_prediction_change_percent=('prediction_change_percent', 'mean'),
        median_prediction_change_percent=('prediction_change_percent', 'median'),
    ).sort_values(['material_type', 'output', 'input', 'requested_input_change_percent']).reset_index(drop=True)

    # Generate sensitivity rankings
    ranking = summary[summary.requested_input_change_percent != 0].groupby(
        ['material_type', 'input', 'output'], as_index=False
    ).agg(
        maximum_absolute_mean_prediction_change_percent=(
            'mean_prediction_change_percent', lambda x: float(np.nanmax(np.abs(x)))),
        mean_absolute_prediction_change_percent=(
            'mean_prediction_change_percent', lambda x: float(np.nanmean(np.abs(x)))),
        maximum_absolute_mean_prediction_change=(
            'mean_prediction_change', lambda x: float(np.nanmax(np.abs(x)))),
    )
    ranking['rank_for_material_output'] = ranking.groupby(
        ['material_type', 'output']
    ).maximum_absolute_mean_prediction_change_percent.rank(method='min', ascending=False).astype(int)
    ranking = ranking.sort_values(['material_type', 'output', 'rank_for_material_output', 'input']).reset_index(drop=True)

    summary.to_csv(d / 'percentage_perturbation_summary_by_material.csv', index=False)
    ranking.to_csv(d / 'percentage_perturbation_input_rankings_by_material.csv', index=False)


    return summary, ranking


def choose_split(df, fraction):
    cut = int(round((1 - fraction) * len(df)))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def choose_internal_validation_split(df, fraction):
    return choose_split(df, fraction)


def main():
    set_seed(SEED)
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    global EDGE_TABLE, EDGE_INDEX, EDGE_ATTR
    EDGE_TABLE, EDGE_INDEX, EDGE_ATTR, _, _ = initialise_knowledge_graph()

    df = clean_numeric_columns(pd.read_csv(CSV, na_values=['', 'NA', 'N/A', 'null', 'None', 'nan']))
    development_raw, test_raw = choose_split(df, TEST_FRACTION)
    selection_train_raw, validation_raw = choose_internal_validation_split(development_raw, VALIDATION_FRACTION)
    device = torch.device(DEVICE)

    train_p, select_pp = fit_preprocessors(selection_train_raw)
    val_p = apply_preprocessors(validation_raw, select_pp)
    selected_epoch, best_val = select_epoch_with_validation(build_dataset(train_p), build_dataset(val_p), TARGET_COLUMNS, device, EPOCHS, LR, BATCH_SIZE, HIDDEN, SEED)

    dev_p, pp = fit_preprocessors(development_raw)
    test_p = apply_preprocessors(test_raw, pp)
    result, model = train_and_evaluate_gat(build_dataset(dev_p), build_dataset(test_p), test_raw, pp, TARGET_COLUMNS, device,
                                           outdir / 'extension_model_results', 'Extension model', selected_epoch, LR, BATCH_SIZE, HIDDEN, SEED,
                                           best_val)

    run_numerical_sensitivity_analysis(
        model, development_raw, test_raw, pp, device, BATCH_SIZE, outdir)
    print(f'Saved results to: {outdir.resolve()}')


if __name__ == '__main__':
    main()
