import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt

with open("./data/output/run_eval_eval.pkl", "rb") as f:
    eval_data = pkl.load(f)
    
with open("./data/output/run_evalpolicy_eval.pkl", "rb") as f:
    policy_data = pkl.load(f)

print("Eval data shape:", eval_data.shape)
print(eval_data.head())

print("\nPolicy data shape:", policy_data.shape)
print(policy_data.head())

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Inside temperature vs time
axes[0].plot(eval_data['Inside Temperatures'], label='Inside Temp', color='red')
axes[0].plot(eval_data['Ambient Temperatures'], label='Ambient Temp', color='blue')
axes[0].axhline(21, color='green', linestyle='--', label='Target (21°C)')
axes[0].set_ylabel('Temperature (°C)')
axes[0].legend()
axes[0].grid()

# Actions și rewards
axes[1].plot(eval_data['Actions'], label='Action', alpha=0.7)
axes[1].plot(eval_data['Rewards'], label='Reward', alpha=0.7)
axes[1].set_xlabel('Time step')
axes[1].set_ylabel('Action / Reward')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig("./data/output/run_eval_plot.png", dpi=150)
plt.show()
# Extrage valoarea din tensor dacă e tensor
import torch

def extract_action(x):
    if isinstance(x, torch.Tensor):
        return int(x.item())
    elif isinstance(x, list) and len(x) > 0:
        if isinstance(x[0], torch.Tensor):
            return int(x[0].item())
        elif isinstance(x[0], list) and isinstance(x[0][0], torch.Tensor):
            return int(x[0][0].item())
    return int(x)

policy_data['Actions'] = policy_data['Actions'].apply(extract_action)
policy_data['Inside Temperatures'] = pd.to_numeric(policy_data['Inside Temperatures'], errors='coerce')
policy_data['Prices'] = pd.to_numeric(policy_data['Prices'], errors='coerce')

import numpy as np
print("Actions dtype:", policy_data['Actions'].dtype)
print("Sample Actions:", policy_data['Actions'].head(10))
print("Unique Actions:", policy_data['Actions'].unique())
policy_data['Actions'] = pd.to_numeric(policy_data['Actions'], errors='coerce')
policy_pivot = policy_data.pivot_table(
    index='Inside Temperatures',
    columns='Prices',
    values='Actions',
    aggfunc='mean'
).fillna(0)  # sau .dropna()
print("Pivot dtype:", policy_pivot.values.dtype)
print("Pivot shape:", policy_pivot.shape)

# pivot table: inside_temp × price, media actions la fiecare ambient_temp
policy_pivot = policy_data.pivot_table(
    index='Inside Temperatures',
    columns='Prices',
    values='Actions',
    aggfunc='mean'
)

plt.figure(figsize=(12, 6))
plt.imshow(policy_pivot, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Mean Action')
plt.xlabel('Price')
plt.ylabel('Inside Temperature')
plt.title('Policy: Action distribution by Inside Temp & Price')
plt.tight_layout()
plt.savefig("./data/output/run_policy_heatmap.png", dpi=150)
plt.show()
