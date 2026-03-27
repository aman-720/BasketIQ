"""
BasketIQ - Module 3: RFM Segmentation & K-Means Clustering
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os, json, warnings
warnings.filterwarnings('ignore')

from config import *

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': FIG_DPI, 'font.size': 11, 'axes.titlesize': 14,
    'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)

# ─── LOAD ─────────────────────────────────────────────────────────────
print("Loading data...")
DTYPES_OP  = {'order_id': np.int32, 'product_id': np.int32,
               'add_to_cart_order': np.int16, 'reordered': np.int8}
DTYPES_ORD = {'order_id': np.int32, 'user_id': np.int32,
               'order_number': np.int16, 'order_dow': np.int8,
               'order_hour_of_day': np.int8}

orders = pd.read_csv(f"{DATA_RAW}/orders.csv", dtype=DTYPES_ORD)
order_products = pd.read_csv(f"{DATA_RAW}/order_products__prior.csv",
                              usecols=['order_id','product_id','reordered'],
                              dtype=DTYPES_OP)
orders_prior = orders[orders.eval_set == 'prior']

# ─── RFM METRICS ─────────────────────────────────────────────────────
print("Computing RFM metrics...")

# Frequency + recency per user (from orders table)
user_orders = orders_prior.groupby('user_id').agg(
    total_orders      = ('order_id', 'count'),
    max_order_num     = ('order_number', 'max'),
    avg_days_between  = ('days_since_prior_order', 'mean'),
    last_days_since   = ('days_since_prior_order', 'last'),
).reset_index()

# Monetary (items) + reorder from order_products
items_per_user = order_products.merge(
    orders_prior[['order_id','user_id']], on='order_id'
).groupby('user_id').agg(
    total_items     = ('product_id', 'count'),
    unique_products = ('product_id', 'nunique'),
    reorder_rate    = ('reordered',  'mean'),
).reset_index()

rfm = user_orders.merge(items_per_user, on='user_id')
rfm['avg_days_between'] = rfm['avg_days_between'].fillna(rfm['avg_days_between'].median())

# RFM scores
rfm['R'] = pd.qcut(rfm['avg_days_between'], RFM_QUANTILES,
                   labels=range(RFM_QUANTILES, 0, -1)).astype(int)
rfm['F'] = pd.qcut(rfm['total_orders'].rank(method='first'), RFM_QUANTILES,
                   labels=range(1, RFM_QUANTILES+1)).astype(int)
rfm['M'] = pd.qcut(rfm['total_items'].rank(method='first'), RFM_QUANTILES,
                   labels=range(1, RFM_QUANTILES+1)).astype(int)
rfm['RFM_Score']   = rfm['R']*100 + rfm['F']*10 + rfm['M']
rfm['RFM_Segment'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
print(f"RFM table: {len(rfm):,} users")

# ─── K-MEANS ─────────────────────────────────────────────────────────
print("\n── K-Means Clustering ──")
features = ['avg_days_between','total_orders','total_items','unique_products','reorder_rate']
X = rfm[features].fillna(rfm[features].median())
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow + silhouette
inertias, silhouettes = [], []
for k in range(2, 11):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_, sample_size=10_000, random_state=42)
    silhouettes.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.3f}")

# Final model
kmeans     = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20, max_iter=500)
rfm['cluster'] = kmeans.fit_predict(X_scaled)
sil_final  = silhouette_score(X_scaled, rfm['cluster'], sample_size=10_000, random_state=42)
print(f"\nFinal silhouette (k={N_CLUSTERS}): {sil_final:.3f}")

# Assign human-readable labels by composite value score
cluster_stats = rfm.groupby('cluster').agg(
    avg_recency      = ('avg_days_between', 'mean'),
    avg_frequency    = ('total_orders',     'mean'),
    avg_monetary     = ('total_items',      'mean'),
    avg_unique_prods = ('unique_products',  'mean'),
    avg_reorder_rate = ('reorder_rate',     'mean'),
    count            = ('user_id',          'count'),
).reset_index()
cluster_stats['composite'] = cluster_stats['avg_frequency'] * cluster_stats['avg_monetary']
cluster_stats = cluster_stats.sort_values('composite', ascending=False).reset_index(drop=True)

label_order = ['Champions','Loyal Customers','Potential Loyalists','At-Risk','Hibernating']
label_map   = {row['cluster']: label_order[i] for i, (_, row) in enumerate(cluster_stats.iterrows())}
rfm['segment']         = rfm['cluster'].map(label_map)
cluster_stats['segment'] = cluster_stats['cluster'].map(label_map)

print("\n── Cluster Profiles ──")
for _, row in cluster_stats.iterrows():
    print(f"  {row['segment']} (n={row['count']:,}): "
          f"orders={row['avg_frequency']:.1f}, items={row['avg_monetary']:.0f}, "
          f"days_between={row['avg_recency']:.1f}, reorder={row['avg_reorder_rate']:.1%}")

# ─── VISUALIZATIONS ───────────────────────────────────────────────────

# Fig 15: Elbow + Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)
K = list(range(2, 11))
ax1.plot(K, inertias, 'o-', color=PALETTE[0], linewidth=2, markersize=6)
ax1.axvline(N_CLUSTERS, color=PALETTE[3], linestyle='--', alpha=0.7, label=f'k={N_CLUSTERS}')
ax1.set_title('Elbow Method'); ax1.set_xlabel('k'); ax1.set_ylabel('Inertia'); ax1.legend()
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax2.plot(K, silhouettes, 'o-', color=PALETTE[2], linewidth=2, markersize=6)
ax2.axvline(N_CLUSTERS, color=PALETTE[3], linestyle='--', alpha=0.7, label=f'k={N_CLUSTERS}')
ax2.set_title('Silhouette Score'); ax2.set_xlabel('k'); ax2.set_ylabel('Score'); ax2.legend()
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/15_elbow_silhouette.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# Fig 16: Segment distribution pie
fig, ax = plt.subplots(figsize=(8, 8))
seg_counts = rfm.segment.value_counts()
wedges, texts, autotexts = ax.pie(
    seg_counts.values, labels=seg_counts.index,
    colors=[PALETTE[i] for i in range(len(seg_counts))],
    autopct='%1.1f%%', startangle=90, pctdistance=0.8,
    wedgeprops=dict(linewidth=2, edgecolor='white'))
for t in autotexts: t.set_fontsize(10); t.set_fontweight('bold')
ax.set_title('Customer Segment Distribution')
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/16_cluster_distribution.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# Fig 17: Radar chart
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
categories = ['Frequency','Monetary','Recency\n(inv)','Product\nVariety','Reorder\nRate']
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist() + [0]
radar_df = cluster_stats[['avg_frequency','avg_monetary','avg_recency',
                           'avg_unique_prods','avg_reorder_rate']].copy()
radar_df['avg_recency'] = radar_df['avg_recency'].max() - radar_df['avg_recency']
for col in radar_df.columns:
    mn, mx = radar_df[col].min(), radar_df[col].max()
    radar_df[col] = (radar_df[col] - mn) / (mx - mn) if mx > mn else 0.5
for i, (_, row) in enumerate(cluster_stats.iterrows()):
    vals = radar_df.iloc[i].tolist() + [radar_df.iloc[i, 0]]
    ax.plot(angles, vals, 'o-', color=PALETTE[i], linewidth=2, label=row['segment'])
    ax.fill(angles, vals, alpha=0.1, color=PALETTE[i])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=10)
ax.set_title('Customer Segment Profiles', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/17_cluster_radar.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# Fig 18: Box plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = [('total_orders','Total Orders'),('total_items','Total Items'),
           ('avg_days_between','Avg Days Between Orders'),('reorder_rate','Reorder Rate')]
for ax, (col, title) in zip(axes.flat, metrics):
    seg_order = [s for s in label_order if s in rfm.segment.values]
    ax.boxplot([rfm[rfm.segment == s][col].dropna() for s in seg_order],
               labels=seg_order, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor='#4C72B033'))
    ax.set_title(title); ax.tick_params(axis='x', rotation=20)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.suptitle('Segment Metric Distributions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/18_cluster_boxplots.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# Fig 19: Frequency vs Monetary scatter (sample 20K points for speed)
fig, ax = plt.subplots(figsize=FIG_SIZE)
rfm_sample = rfm.sample(min(20_000, len(rfm)), random_state=42)
for i, seg in enumerate(label_order):
    sub = rfm_sample[rfm_sample.segment == seg]
    if len(sub): ax.scatter(sub.total_orders, sub.total_items, alpha=0.3, s=8,
                             color=PALETTE[i], label=seg)
ax.set_title('Frequency vs Monetary by Segment')
ax.set_xlabel('Total Orders'); ax.set_ylabel('Total Items Purchased')
ax.legend(markerscale=3, fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/19_frequency_vs_monetary.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# Fig 20: RFM score distributions
fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE_WIDE)
for ax, col, title, color in zip(axes, ['R','F','M'],
                                  ['Recency','Frequency','Monetary'], PALETTE):
    rfm[col].value_counts().sort_index().plot(kind='bar', ax=ax, color=color, edgecolor='white')
    ax.set_title(f'{title} Score Distribution')
    ax.set_xlabel('Score'); ax.set_ylabel('Users')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.suptitle('RFM Score Distributions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/20_rfm_distributions.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ─── SAVE ─────────────────────────────────────────────────────────────
rfm.to_csv(f"{DATA_PROCESSED}/rfm_segments.csv", index=False)
cluster_stats.to_csv(f"{DATA_PROCESSED}/cluster_profiles.csv", index=False)

rfm_summary = {
    "total_users": int(len(rfm)),
    "n_clusters": N_CLUSTERS,
    "silhouette_score": float(sil_final),
    "segments": {
        row['segment']: {
            "count": int(row['count']),
            "pct": float(round(row['count']/len(rfm)*100, 1)),
            "avg_orders": float(round(row['avg_frequency'], 1)),
            "avg_items": float(round(row['avg_monetary'], 1)),
            "avg_days_between": float(round(row['avg_recency'], 1)),
            "avg_reorder_rate": float(round(row['avg_reorder_rate'], 3)),
        } for _, row in cluster_stats.iterrows()
    }
}
with open(f"{DATA_PROCESSED}/rfm_summary.json", 'w') as f:
    json.dump(rfm_summary, f, indent=2)

print(f"\n✓ RFM segmentation complete. 6 visualizations generated.")
