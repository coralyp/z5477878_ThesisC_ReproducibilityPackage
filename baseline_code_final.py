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
CSV = next(Path(__file__).resolve().parents[2].rglob('dataset.csv'))
OUTDIR = 'results'
EPOCHS, LR, HIDDEN, BATCH_SIZE, SEED = 150,.01, 8, 32, 42
DEVICE = 'cpu'
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE = 'bolt://127.0.0.1:7687', 'neo4j', 'neo4j_PASSWORD', 'neo4j'
TEST_FRACTION = VALIDATION_FRACTION =.10
UNKNOWN_CATEGORY = '__UNKNOWN__'


# Define information for KG
INPUT_COLUMNS = ['material_type', 'material_form', 'material_diameter', 'laser_power', 'scanning_speed', 'hatching_distance', 'layer_thickness', 'rotation_angle']
TARGET_COLUMNS = ['hardness', 'yield_stress', 'tensile_strength', 'elongation_to_failure']
CATEGORICAL_COLUMNS = ['material_type', 'material_form']
NUMERICAL_INPUT_COLUMNS = [c for c in INPUT_COLUMNS if c not in CATEGORICAL_COLUMNS]
DATA_COLUMNS = INPUT_COLUMNS + TARGET_COLUMNS
NODE_ORDER = INPUT_COLUMNS + TARGET_COLUMNS
NODE_INDEX = {n: i for i, n in enumerate(NODE_ORDER)}
NUM_NODES = len(NODE_ORDER)
NODE_FEATURE_DIM = NUM_NODES
TARGET_INDICES = [NODE_INDEX[c] for c in TARGET_COLUMNS]

EDGE_INDEX = None


# Set random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Setup, load, and initialise KG from Neo4j
def edge_table_to_edge_index(df):
    return torch.tensor(df[['source_index', 'target_index']].to_numpy().T, dtype=torch.long).contiguous()

def load_edge_table_from_neo4j(uri, user, password, database):
    node_q = '''MATCH (n:LPBF_Node) WHERE n.code_name IN $nodes RETURN n.code_name AS code_name ORDER BY code_name'''
    edge_q = '''MATCH (s:LPBF_Node)-[:DIRECT_INFLUENCE]->(t:LPBF_Node)
                WHERE s.code_name IN $nodes AND t.code_name IN $nodes
                RETURN s.code_name AS source, t.code_name AS target
                ORDER BY s.code_name, t.code_name'''

    with GraphDatabase.driver(uri, auth=(user, password)) as driver, driver.session(database=database) as session:
        nodes = pd.DataFrame(dict(r) for r in session.run(node_q, nodes=NODE_ORDER))
        edges = pd.DataFrame(dict(r) for r in session.run(edge_q, nodes=NODE_ORDER))

    edges = edges.drop_duplicates(['source', 'target']).sort_values(['source', 'target'], kind='mergesort').reset_index(drop=True)
    edges['source_index'] = edges.source.map(NODE_INDEX).astype(int)
    edges['target_index'] = edges.target.map(NODE_INDEX).astype(int)
    return edges, nodes.drop_duplicates('code_name').reset_index(drop=True)

def initialise_knowledge_graph():
    edges, nodes = load_edge_table_from_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    edge_index = edge_table_to_edge_index(edges)
    return edges, edge_index, nodes


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


# Clean build_dataset and _encode_categories categorical inputs
def clean_numeric_columns(df):
    df = df.copy()
    for c in NUMERICAL_INPUT_COLUMNS + TARGET_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].replace(r'^\s*$', np.nan, regex=True)
    return (df)

def _fit_label_encoder_with_unknown(s):
    e = LabelEncoder()
    e.classes_ = np.asarray(sorted(set(s.astype(str)) | {UNKNOWN_CATEGORY}), dtype=object)
    return e

def _encode_categories(df, encoders):
    df = df.copy()
    for c, e in encoders.items():
        v = df[c].astype(str)
        df[c] = e.transform(v.where(v.isin(e.classes_), UNKNOWN_CATEGORY))
    return df


# Fit preprocessing information (encoders, standardisations)
def fit_preprocessors(df, target_columns=None):
    targets = list(target_columns or TARGET_COLUMNS)
    encoders = {c: _fit_label_encoder_with_unknown(df[c]) for c in CATEGORICAL_COLUMNS}
    df = _encode_categories(df.copy(), encoders)

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

    # Assign data entry values to nodes
    for n in INPUT_COLUMNS:
        x[NODE_INDEX[n], NODE_INDEX[n]] = float(row[n])

    # Create target mask; assign True if target value is known, False if not
    vals = np.asarray([row[t] for t in targets], np.float32,)
    mask = np.isfinite(vals)

    # Create PyTorch graph
    return Data(x=x, edge_index=EDGE_INDEX.clone(), y=torch.from_numpy(np.nan_to_num(vals)).unsqueeze(0), y_mask=torch.from_numpy(mask).unsqueeze(0))


# Create graph for each data entry
def build_dataset(df, target_columns=None):
    return [row_to_graph(row, target_columns) for _, row in df.iterrows()]


# GAT Architecture
class GATRegressor(nn.Module):
    def __init__(self, hidden, targets):
        super().__init__()
        self.output_target_indices = [NODE_INDEX[t] for t in targets]

        self.g1 = GATConv(NODE_FEATURE_DIM, hidden, heads=4, dropout=0,) # Initialise GAT Layer 1
        self.g2 = GATConv(hidden * 4, hidden, heads=1, dropout=0,) # Initialise GAT Layer 2
        self.fc = nn.Linear(hidden * len(targets), len(targets),) # Define fully connected layer

    # Forward pass
    def forward(self, x, edge_index, batch):
        # Run each GAT layer + ELU activation
        x = F.elu(self.g1(x, edge_index))
        x = F.elu(self.g2(x, edge_index)).view(-1, NUM_NODES, self.g2.out_channels,)

        # Keep ONLY target nodes and concatenate them
        # Fully connected layer
        return self.fc(x[:, self.output_target_indices].flatten(1))


# Calculate MSE loss only if target value exists in build_dataset
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
        loss = masked_mse_loss(model(batch.x, batch.edge_index, batch.batch), batch.y, batch.y_mask)
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
        p.append(model(batch.x, batch.edge_index, batch.batch).cpu().numpy())
        y.append(batch.y.cpu().numpy())
        m.append(batch.y_mask.cpu().numpy().astype(bool))

    return np.vstack(p), np.vstack(y), np.vstack(m)


# Calculate performance metrics
def compute_metrics_1d(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    n = len(y_true)
    if not n:
        return dict(observed_samples=0, MAE=np.nan, MSE=np.nan, RMSE=np.nan, R2=np.nan)
    mse = mean_squared_error(y_true, y_pred)
    return dict(observed_samples=n, MAE=float(mean_absolute_error(y_true, y_pred)), MSE=float(mse), RMSE=float(math.sqrt(mse)), R2=float(r2_score(y_true, y_pred)) if n > 1 else np.nan)

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
        ape = np.full_like(actual, np.nan, dtype=float)
        ape[valid] = ae[valid] / np.abs(actual[valid]) * 100
        acc = np.clip(100 - ape, 0, 100)
        mse = mean_squared_error(actual, pred)
        rows.append(dict(property=name, test_samples=len(actual), valid_percentage_error_samples=int(valid.sum()), mean_actual=float(actual.mean()), 
                         mean_predicted=float(pred.mean()), MAE=float(mean_absolute_error(actual, pred)), RMSE=float(math.sqrt(mse)), 
                         R2=float(r2_score(actual, pred)) if len(actual) > 1 else np.nan, MAPE_percent=float(np.nanmean(ape)) if valid.any() else np.nan, 
                         median_absolute_percentage_error_percent=float(np.nanmedian(ape)) if valid.any() else np.nan, mean_prediction_accuracy_percent=float(np.nanmean(acc)) if valid.any() else np.nan, 
                         median_prediction_accuracy_percent=float(np.nanmedian(acc)) if valid.any() else np.nan, 
                         within_5_percent_error=float(np.nanmean(ape <= 5) * 100) if valid.any() else np.nan, 
                         within_10_percent_error=float(np.nanmean(ape <= 10) * 100) if valid.any() else np.nan, 
                         within_20_percent_error=float(np.nanmean(ape <= 20) * 100) if valid.any() else np.nan))
    return pd.DataFrame(rows)

def evaluate_scaled_mse(model, loader, device, output_dim):
    pred, y, mask = predict(model, loader, device)
    mses = [mean_squared_error(y[m, i], pred[m, i]) for i in range(output_dim) for m in [mask[:, i].astype(bool)] if m.any()]
    return float(np.mean(mses)) if mses else float('inf')


# Select optimal epoch with lowest MSE between validation and training
def select_epoch_with_validation(train_ds, validation_ds, targets, device, epochs, lr, batch_size, hidden_dim, seed):
    set_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_ds, batch_size=batch_size)
    model = GATRegressor(hidden_dim, targets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vals = []
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, device)
        vals.append(evaluate_scaled_mse(model, val_loader, device, len(targets)))
    best_epoch = int(np.argmin(vals)) + 1
    return best_epoch, float(vals[best_epoch - 1])


# Retrain final model with optimal number of epochs
def train_and_evaluate_gat(development_ds, test_ds, test_raw, pp, targets, device, model_dir, model_name, selected_epochs, lr, batch_size, hidden_dim, seed, best_validation_mse):
    set_seed(seed)
    model_dir.mkdir(parents=True, exist_ok=True)

    dev_loader = DataLoader(development_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = GATRegressor(hidden_dim, targets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(selected_epochs):
        train_one_epoch(model, dev_loader, optimizer, device)

    pred_s, y_s, mask = predict(model, test_loader, device)
    pred = pp.target_scaler.inverse_transform(pred_s)
    y = np.where(mask, pp.target_scaler.inverse_transform(y_s), np.nan)

    overall = compute_metrics_1d(y[mask], pred[mask])
    by_target = per_target_metrics(y, pred, mask, targets)
    acc_df = compute_prediction_accuracy_by_target(y, pred, mask, targets)

    pd.DataFrame([{'property': n, **v} for n, v in by_target.items()]).to_csv(model_dir / 'per_target_metrics.csv', index=False)
    acc_df.to_csv(model_dir / 'per_target_prediction_accuracy.csv', index=False)
    pd.DataFrame([{'model': model_name, 'selected_epoch': selected_epochs, 'best_validation_mse_scaled': best_validation_mse, **{f'overall_{k}': v for k, v in overall.items() if k != 'observed_samples'}}]).to_csv(model_dir / 'model_summary.csv', index=False)

    return by_target


def choose_internal_validation_split(df, fraction):
    return choose_split(df, fraction)


def choose_split(df, fraction):
    cut = int(round((1 - fraction) * len(df)))

    return (df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True),)


def main():
    set_seed(SEED)
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True,)

    global EDGE_INDEX
    _, EDGE_INDEX, _ = initialise_knowledge_graph()

    df = clean_numeric_columns(pd.read_csv(CSV, na_values=['', 'NA', 'N/A', 'null', 'None', 'nan'],))
    development_raw, test_raw = choose_split(df, TEST_FRACTION)
    selection_train_raw, validation_raw = choose_internal_validation_split(development_raw, VALIDATION_FRACTION)
    device = torch.device(DEVICE)

    train_p, select_pp = fit_preprocessors(selection_train_raw)
    val_p = apply_preprocessors(validation_raw, select_pp)
    selected_epoch, best_val = select_epoch_with_validation(build_dataset(train_p), build_dataset(val_p), TARGET_COLUMNS, device, EPOCHS, LR, BATCH_SIZE, HIDDEN, SEED)

    dev_p, pp = fit_preprocessors(development_raw)
    test_p = apply_preprocessors(test_raw, pp)

    results = train_and_evaluate_gat(build_dataset(dev_p), build_dataset(test_p), test_raw, pp, TARGET_COLUMNS, device, outdir / 'baseline_model_results', 'Baseline model', selected_epoch, LR, BATCH_SIZE, HIDDEN, SEED, best_val)

    print(f'Saved results to: {outdir.resolve()}')


if __name__ == '__main__':
    main()
