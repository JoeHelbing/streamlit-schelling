import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the simulation results
with open("schelling_results.json", "r") as f:
    results = json.load(f)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(results)

# Get unique values for x and y axes
empty_ratios = sorted(df["empty_ratio"].unique())
similarity_thresholds = sorted(df["similarity_threshold"].unique())

# Create a pivot table for visualization
pivot_df = df.pivot_table(
    values="mean_similarity", index="empty_ratio", columns="similarity_threshold"
)

# Create subplots: one 3D surface and one 2D heatmap
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "surface"}, {"type": "heatmap"}]],
    subplot_titles=("3D Surface Plot", "Heatmap View"),
    horizontal_spacing=0.05,
)

# Add 3D surface plot
fig.add_trace(
    go.Surface(
        z=pivot_df.values,
        x=pivot_df.columns,  # similarity_threshold
        y=pivot_df.index,  # empty_ratio
        colorscale="Viridis",
        colorbar=dict(x=0.45, title="Mean Similarity"),
        hovertemplate="Empty Ratio: %{y:.2f}<br>Similarity Threshold: %{x:.2f}<br>Mean Similarity: %{z:.4f}<extra></extra>",
    ),
    row=1,
    col=1,
)

# Add heatmap plot
fig.add_trace(
    go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,  # similarity_threshold
        y=pivot_df.index,  # empty_ratio
        colorscale="Viridis",
        colorbar=dict(title="Mean Similarity"),
        hovertemplate="Empty Ratio: %{y:.2f}<br>Similarity Threshold: %{x:.2f}<br>Mean Similarity: %{z:.4f}<extra></extra>",
    ),
    row=1,
    col=2,
)

# Update layout with better titles and spacing
fig.update_layout(
    title="Schelling Model - Effect of Empty Ratio and Similarity Threshold",
    scene=dict(
        xaxis_title="Similarity Threshold",
        yaxis_title="Empty Ratio",
        zaxis_title="Mean Similarity",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    ),
    xaxis_title="Similarity Threshold",
    yaxis_title="Empty Ratio",
    height=800,
    width=1200,
    margin=dict(l=65, r=50, b=65, t=90),
)

# Save the interactive plot as HTML
fig.write_html("schelling_visualization.html")
print("Visualization saved as 'schelling_visualization.html'")

# Optional - create just the 3D surface in a separate file
fig_3d = go.Figure(
    data=[
        go.Surface(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale="Viridis",
            hovertemplate="Empty Ratio: %{y:.2f}<br>Similarity Threshold: %{x:.2f}<br>Mean Similarity: %{z:.4f}<extra></extra>",
        )
    ]
)

fig_3d.update_layout(
    title="Schelling Model - 3D Visualization",
    scene=dict(
        xaxis_title="Similarity Threshold",
        yaxis_title="Empty Ratio",
        zaxis_title="Mean Similarity",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    ),
    height=800,
    width=1000,
    margin=dict(l=65, r=50, b=65, t=90),
)

fig_3d.write_html("schelling_3d_plot.html")
print("3D plot saved as 'schelling_3d_plot.html'")
