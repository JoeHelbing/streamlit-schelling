import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split,  # Still needed for initial half split
)
from sklearn.neighbors import NearestNeighbors  # Needed for LHS mapping
from xgboost import XGBRegressor
from scipy.interpolate import griddata  # Ensure this import remains
import plotly.graph_objects as go

# load data
df = pd.read_json("data/schelling_res_var_outcome.json")

# features and target names
features = ["empty_ratio", "similarity_threshold"]
target = "mean_similarity"


# --- Keep LHS function ---
def lhs(n_samples, n_dims, random_state=None):
    rng = np.random.default_rng(random_state)
    H = np.zeros((n_samples, n_dims))
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        H[:, j] = (perm + rng.random(n_samples)) / n_samples
    return H


# --- End LHS function ---

# Split data into two random halves (indices)
indices = np.arange(len(df))
indices_a, indices_b = train_test_split(indices, test_size=0.5, random_state=42)

# Create the half dataframes using the indices
df_half_a = (
    df.iloc[indices_a].copy().reset_index(drop=True)
)  # Reset index for easier iloc later
df_half_b = df.iloc[indices_b].copy().reset_index(drop=True)  # Reset index

halves_data = {
    "A": {"train_eval": df_half_a, "predict_on": df_half_b},
    "B": {"train_eval": df_half_b, "predict_on": df_half_a},
}

train_percentages = [0.25, 0.50, 0.75]

base_eval_dir = os.path.join(
    "evaluations", "xgboost_half_split_lhs"
)  # New base dir name
os.makedirs(base_eval_dir, exist_ok=True)

sns.set_theme(style="whitegrid")

results_summary = []

for half_label, data_map in halves_data.items():
    df_current_half = data_map["train_eval"]
    df_other_half = data_map["predict_on"]

    # Features and target for the *other* half (used for final prediction)
    X_other = df_other_half[features]
    y_other = df_other_half[target]

    # Features and target for the *current* half (used for LHS and training/testing)
    X_current = df_current_half[features]
    y_current = df_current_half[target]

    for train_perc in train_percentages:
        run_label = f"trained_on_half_{half_label}_lhs_train_{int(train_perc * 100)}pct"
        current_eval_dir = os.path.join(base_eval_dir, run_label)
        os.makedirs(current_eval_dir, exist_ok=True)

        print(f"\n--- Running: {run_label} ---")

        # --- Use LHS for Train/Test Split within the current half ---
        n_total_current = len(df_current_half)
        n_train_samples = int(train_perc * n_total_current)

        # Generate LHS design points in normalized space
        design = lhs(
            n_train_samples, X_current.shape[1], random_state=42
        )  # Use fixed random state for reproducibility

        # Scale design points to the range of features in the *current* half
        mins_current = X_current.min().values
        maxs_current = X_current.max().values
        scaled_design = design * (maxs_current - mins_current) + mins_current

        # Find nearest actual data points in the *current* half to the LHS design points
        nn = NearestNeighbors(n_neighbors=1).fit(X_current.values)
        dists, idxs = nn.kneighbors(scaled_design)
        train_idx_current = np.unique(
            idxs.flatten()
        )  # Indices relative to df_current_half

        # Ensure we don't accidentally get more train samples than requested due to duplicates
        if len(train_idx_current) > n_train_samples:
            # If more points found than requested (unlikely with n_neighbors=1 but possible), randomly sample
            rng_select = np.random.default_rng(random_state=43)  # Different seed
            train_idx_current = rng_select.choice(
                train_idx_current, n_train_samples, replace=False
            )

        # Define test indices as those *not* in the training set within the current half
        all_idx_current = np.arange(n_total_current)
        test_idx_current = np.setdiff1d(all_idx_current, train_idx_current)

        # Create train/test sets using indices relative to the current half DataFrame
        X_train = X_current.iloc[train_idx_current]
        y_train = y_current.iloc[train_idx_current]
        X_test = X_current.iloc[test_idx_current]
        y_test = y_current.iloc[test_idx_current]
        # --- End LHS Split ---

        # Configure XGBoost (CPU only)
        tree_method = "hist"
        # predictor = "cpu_predictor" # Often not needed
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            tree_method=tree_method,
            n_jobs=-1,
            eval_metric="rmse",
            random_state=42,
        )

        # Train the model
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,  # Keep verbose False for cleaner loop output
        )
        evals_result = model.evals_result()

        # Evaluate on the test set (from the *same* half)
        preds_test = model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
        r2_test = r2_score(y_test, preds_test)

        # Evaluate on the training set (from the *same* half)
        preds_train = model.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
        r2_train = r2_score(y_train, preds_train)

        print(
            f"  Test Set (Half {half_label}, LHS): RMSE={rmse_test:.4f}, R2={r2_test:.4f}"
        )
        print(
            f"  Train Set (Half {half_label}, LHS): RMSE={rmse_train:.4f}, R2={r2_train:.4f}"
        )

        # Predict on the *other* half
        preds_other_half = model.predict(X_other)
        rmse_other_half = np.sqrt(mean_squared_error(y_other, preds_other_half))
        r2_other_half = r2_score(y_other, preds_other_half)
        print(
            f"  Prediction on Half {('B' if half_label == 'A' else 'A')}): RMSE={rmse_other_half:.4f}, R2={r2_other_half:.4f}"
        )

        # --- Save results for this run ---
        # 1. Metrics
        metrics = {
            "run_label": run_label,
            "trained_on_half": half_label,
            "train_percentage": train_perc,
            "sampling_method": "LHS",  # Added sampling method
            "n_train_samples_actual": len(train_idx_current),  # Actual number used
            "n_test_samples_actual": len(test_idx_current),
            "rmse_test_same_half": float(rmse_test),
            "r2_test_same_half": float(r2_test),
            "rmse_train_same_half": float(rmse_train),
            "r2_train_same_half": float(r2_train),
            "rmse_pred_other_half": float(rmse_other_half),
            "r2_pred_other_half": float(r2_other_half),
        }
        results_summary.append(metrics)
        with open(os.path.join(current_eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # 2. Training History
        serializable_evals = {
            outer_key: {
                inner_key: [float(val) for val in inner_list]
                for inner_key, inner_list in outer_val.items()
            }
            for outer_key, outer_val in evals_result.items()
        }
        with open(os.path.join(current_eval_dir, "evals_result.json"), "w") as f:
            json.dump(serializable_evals, f, indent=2)

        # 3. Predictions on the other half
        predictions_df = pd.DataFrame(
            {
                "true_mean_similarity": y_other,
                "predicted_mean_similarity": preds_other_half,
            }
        )
        predictions_df.to_csv(
            os.path.join(current_eval_dir, "predictions_on_other_half.csv"), index=False
        )

        # 4. Plots (Save only, no show)
        #   a) True vs Predicted (Test Set - Same Half)
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=preds_test, alpha=0.6, s=50)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            linewidth=2,
        )
        plt.xlabel(f"True mean_similarity (Test Set - Half {half_label})")
        plt.ylabel("Predicted mean_similarity")
        plt.title(f"True vs Predicted (Test Set - {run_label})")
        plt.savefig(os.path.join(current_eval_dir, "true_vs_pred_test_set.png"))
        plt.close()

        #   b) Residuals (Test Set - Same Half)
        residuals_test = y_test - preds_test
        plt.figure()
        sns.histplot(residuals_test, bins=30, kde=True)
        plt.xlabel("Residual (True - Predicted)")
        plt.title(f"Residuals Distribution (Test Set - {run_label})")
        plt.savefig(os.path.join(current_eval_dir, "residuals_hist_test_set.png"))
        plt.close()

        #   c) Feature Importance
        plt.figure()
        importances = model.feature_importances_
        sns.barplot(x=features, y=importances)
        plt.title(f"Feature Importances ({run_label})")
        plt.savefig(os.path.join(current_eval_dir, "feature_importance.png"))
        plt.close()

        # Combine all features and true/predicted values
        X_all = df[features]
        y_true_all = df[target]
        # Predict on entire dataset with current model
        all_preds = model.predict(X_all)

        # Create grid for surface plotting
        xi = np.linspace(X_all[features[0]].min(), X_all[features[0]].max(), 100)
        yi = np.linspace(X_all[features[1]].min(), X_all[features[1]].max(), 100)
        XI, YI = np.meshgrid(xi, yi)

        # Interpolate true and predicted onto grid
        Z_true = griddata(
            (X_all[features[0]], X_all[features[1]]),
            y_true_all,
            (XI, YI),
            method='cubic'
        )
        Z_pred = griddata(
            (X_all[features[0]], X_all[features[1]]),
            all_preds,
            (XI, YI),
            method='cubic'
        )

        # Interactive Overlay True vs Predicted Surface using Plotly
        # Create grid for surface plotting (reuse XI, YI from above)
        # Note: XI, YI, Z_true, Z_pred already computed
        fig = go.Figure()
        # True values surface
        fig.add_trace(
            go.Surface(
                x=XI,
                y=YI,
                z=Z_true,
                colorscale='Viridis',
                opacity=0.7,
                name='True Surface'
            )
        )
        # Predicted values surface
        fig.add_trace(
            go.Surface(
                x=XI,
                y=YI,
                z=Z_pred,
                colorscale='Plasma',
                opacity=0.3,
                name='Predicted Surface'
            )
        )
        fig.update_layout(
            title=f"True vs Predicted Surface Overlay ({run_label})",
            scene=dict(
                xaxis_title=features[0],
                yaxis_title=features[1],
                zaxis_title=target
            )
        )
        # Save interactive HTML
        fig.write_html(os.path.join(current_eval_dir, "true_vs_pred_surface_overlay.html"))
        # Compute and plot absolute error surface
        error_surface = np.abs(Z_pred - Z_true)
        fig_error = go.Figure()
        fig_error.add_trace(
            go.Surface(
                x=XI,
                y=YI,
                z=error_surface,
                colorscale='YlOrRd',
                opacity=0.8,
                name='Absolute Error Surface'
            )
        )
        fig_error.update_layout(
            title=f"Absolute Error Surface ({run_label})",
            scene=dict(
                xaxis_title=features[0],
                yaxis_title=features[1],
                zaxis_title='Absolute Error'
            )
        )
        # Save interactive error surface HTML
        fig_error.write_html(os.path.join(current_eval_dir, "absolute_error_surface.html"))


# Save summary results
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(os.path.join(base_eval_dir, "summary_results.csv"), index=False)
print(
    f"\nSummary results saved to: {os.path.join(base_eval_dir, 'summary_results.csv')}"
)
