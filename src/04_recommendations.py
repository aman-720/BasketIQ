"""
BasketIQ - Module 4: Product Recommendation Engine
Three methods: Item-Item CF, Co-purchase, Reorder Probability
Memory-optimised: never loads full 32M-row merged DataFrame.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import os, json, warnings, gc
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
               'order_number': np.int16}

orders        = pd.read_csv(f"{DATA_RAW}/orders.csv", dtype=DTYPES_ORD,
                             usecols=['order_id','user_id','eval_set'])
order_products = pd.read_csv(f"{DATA_RAW}/order_products__prior.csv",
                              usecols=['order_id','product_id','reordered'],
                              dtype=DTYPES_OP)
products      = pd.read_csv(f"{DATA_RAW}/products.csv",
                             usecols=['product_id','product_name','aisle_id'])
aisles        = pd.read_csv(f"{DATA_RAW}/aisles.csv")

products_full = products.merge(aisles, on='aisle_id')
product_names = dict(zip(products_full.product_id, products_full.product_name))
del products, aisles; gc.collect()

orders_prior  = orders[orders.eval_set == 'prior'][['order_id','user_id']].copy()
# Build a fast order_id → user_id map (no 32M-row merge)
order_to_user = orders_prior.set_index('order_id')['user_id']
del orders; gc.collect()

print(f"Loaded {len(order_products):,} transactions")

# ═══════════════════════════════════════════════════════════════════════
# METHOD 1: Item-Item Collaborative Filtering (top 150 products)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Item-Item Collaborative Filtering ──")
top150   = order_products.product_id.value_counts().head(150).index.tolist()
op_top   = order_products[order_products.product_id.isin(top150)].copy()
# Add user_id via fast map — operates only on the filtered subset
op_top['user_id'] = op_top['order_id'].map(order_to_user)
op_top.dropna(subset=['user_id'], inplace=True)
op_top['user_id'] = op_top['user_id'].astype(np.int32)

user_product = op_top.groupby(['user_id','product_id']).size().reset_index(name='count')
user_product['count'] = user_product['count'].clip(upper=10)
matrix   = user_product.pivot_table(index='user_id', columns='product_id',
                                     values='count', fill_value=0)
print(f"User-Product matrix: {matrix.shape}")

sparse_m    = csr_matrix(matrix.values.T)
item_sim    = cosine_similarity(sparse_m)
item_sim_df = pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)
del sparse_m, matrix, user_product; gc.collect()

recommendations = {}
for pid in item_sim_df.columns:
    sim_scores = item_sim_df[pid].drop(pid).sort_values(ascending=False).head(10)
    recommendations[pid] = {
        'product_name': product_names.get(pid, str(pid)),
        'similar_products': [
            {'product_id': int(s), 'product_name': product_names.get(s, str(s)),
             'similarity': float(round(item_sim_df.loc[pid, s], 4))}
            for s in sim_scores.index
        ]
    }

# ═══════════════════════════════════════════════════════════════════════
# METHOD 2: Frequently Bought Together
# ═══════════════════════════════════════════════════════════════════════
print("\n── Frequently Bought Together ──")
sampled_oids = op_top.order_id.drop_duplicates().sample(
    min(100_000, op_top.order_id.nunique()), random_state=42)
op_s = op_top[op_top.order_id.isin(sampled_oids)][['order_id','product_id']].copy()
op_pairs = op_s.merge(op_s, on='order_id', suffixes=('_a','_b'))
op_pairs = op_pairs[op_pairs.product_id_a < op_pairs.product_id_b]
del op_s; gc.collect()

pair_counts = op_pairs.groupby(['product_id_a','product_id_b']).size() \
    .reset_index(name='co_count').sort_values('co_count', ascending=False)
del op_pairs; gc.collect()

pair_counts = pair_counts \
    .merge(products_full[['product_id','product_name']], left_on='product_id_a', right_on='product_id') \
    .rename(columns={'product_name':'product_a'}).drop('product_id', axis=1) \
    .merge(products_full[['product_id','product_name']], left_on='product_id_b', right_on='product_id') \
    .rename(columns={'product_name':'product_b'}).drop('product_id', axis=1)

print(f"Product pairs: {len(pair_counts):,}")
print("\nTop 15 Frequently Bought Together:")
for _, row in pair_counts.head(15).iterrows():
    print(f"  {row['product_a']} + {row['product_b']} ({row['co_count']:,} times)")

# ═══════════════════════════════════════════════════════════════════════
# METHOD 3: Reorder Probability (next-basket prediction)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Next-Basket Prediction (Reorder Probability) ──")
# Work only on op_top (top-150 products) to avoid 32M-row merge
user_total_orders = orders_prior.groupby('user_id')['order_id'].count().reset_index(name='total_orders')

user_prod_freq = op_top.groupby(['user_id','product_id']) \
    .agg(times_ordered=('order_id','count')).reset_index()
user_prod_freq = user_prod_freq.merge(user_total_orders, on='user_id')
user_prod_freq['reorder_prob'] = user_prod_freq['times_ordered'] / user_prod_freq['total_orders']
user_prod_freq['product_name'] = user_prod_freq['product_id'].map(product_names)
del op_top; gc.collect()

sample_uid = user_prod_freq.groupby('user_id').size().sort_values(ascending=False).index[0]
user_recs  = user_prod_freq[user_prod_freq.user_id == sample_uid] \
    .sort_values('reorder_prob', ascending=False).head(10)
print(f"\nSample next-basket for User {sample_uid}:")
for _, row in user_recs.iterrows():
    print(f"  {row['product_name']}: {row['reorder_prob']:.0%} "
          f"({row['times_ordered']}/{row['total_orders']} orders)")

# ─── VISUALIZATIONS ───────────────────────────────────────────────────

# Fig 21: Frequently Bought Together
fig, ax = plt.subplots(figsize=(12, 9))
top20_pairs = pair_counts.head(20).copy()
top20_pairs['pair'] = top20_pairs['product_a'] + '\n+ ' + top20_pairs['product_b']
bars = ax.barh(range(len(top20_pairs)), top20_pairs['co_count'].values,
               color=PALETTE[0], alpha=0.85)
ax.set_yticks(range(len(top20_pairs))); ax.set_yticklabels(top20_pairs['pair'].values, fontsize=8)
ax.invert_yaxis(); ax.set_title('Top 20 Frequently Bought Together'); ax.set_xlabel('Co-purchase Count')
for bar in bars:
    ax.text(bar.get_width()+200, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():,.0f}', ha='left', va='center', fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/21_frequently_bought_together.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# Fig 22: Product similarity heatmap (top 20)
fig, ax = plt.subplots(figsize=(14, 12))
top20_pids = [p for p in order_products.product_id.value_counts().head(20).index.tolist()
              if p in item_sim_df.columns]
sim_sub = item_sim_df.loc[top20_pids, top20_pids].copy()
name_map = {p: product_names.get(p, str(p))[:25] for p in top20_pids}
sim_sub.index   = [name_map[p] for p in sim_sub.index]
sim_sub.columns = [name_map[p] for p in sim_sub.columns]
mask = np.triu(np.ones_like(sim_sub, dtype=bool), k=0)
sns.heatmap(sim_sub, mask=mask, cmap='YlOrRd', annot=True, fmt='.2f', ax=ax,
            linewidths=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Cosine Similarity'})
ax.set_title('Product Similarity Matrix (Top 20 Products)')
plt.xticks(rotation=45, ha='right', fontsize=8); plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/22_product_similarity.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# Fig 23: Reorder probability distribution
fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
ax.hist(user_prod_freq.reorder_prob, bins=50, color=PALETTE[2], edgecolor='white', alpha=0.85)
ax.set_title('Distribution of Product Reorder Probabilities')
ax.set_xlabel('Reorder Probability'); ax.set_ylabel('Count')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/23_reorder_prob_dist.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ─── SAVE ─────────────────────────────────────────────────────────────
pair_counts.head(100).to_csv(f"{DATA_PROCESSED}/frequently_bought_together.csv", index=False)
with open(f"{DATA_PROCESSED}/product_recommendations.json", 'w') as f:
    json.dump({str(k): v for k, v in list(recommendations.items())[:50]}, f, indent=2)

rec_summary = {
    "total_product_pairs": int(len(pair_counts)),
    "top_pair": f"{pair_counts.iloc[0]['product_a']} + {pair_counts.iloc[0]['product_b']}",
    "top_pair_count": int(pair_counts.iloc[0]['co_count']),
    "methods": ["Item-Item Collaborative Filtering", "Co-Purchase Analysis", "Reorder Probability"],
}
with open(f"{DATA_PROCESSED}/recommendation_summary.json", 'w') as f:
    json.dump(rec_summary, f, indent=2)

print(f"\n✓ Recommendation engine complete. 3 visualizations generated.")
