"""
BasketIQ - Module 2: Association Rules Mining
Mines frequent itemsets via Apriori at aisle and product level.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os, json, warnings, time, gc
warnings.filterwarnings('ignore')

from config import *

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': FIG_DPI, 'font.size': 11, 'axes.titlesize': 14,
    'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)

t0 = time.time()

# ─── LOAD (minimal columns only) ─────────────────────────────────────
print("Loading data...")
DTYPES_OP = {'order_id': np.int32, 'product_id': np.int32,
             'add_to_cart_order': np.int16, 'reordered': np.int8}
op = pd.read_csv(f"{DATA_RAW}/order_products__prior.csv",
                 usecols=['order_id','product_id'], dtype=DTYPES_OP)
products = pd.read_csv(f"{DATA_RAW}/products.csv",
                       usecols=['product_id','product_name','aisle_id'])
aisles   = pd.read_csv(f"{DATA_RAW}/aisles.csv")
op = op.merge(products, on='product_id').merge(aisles, on='aisle_id')
print(f"Loaded {len(op):,} rows | {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════
# AISLE-LEVEL RULES  (sample 50K orders — robust for real data)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Aisle-Level Association Rules ──")
sampled_oids = op.order_id.drop_duplicates().sample(50_000, random_state=42)
op_s = op[op.order_id.isin(sampled_oids)]
baskets = op_s.groupby('order_id')['aisle'].apply(lambda x: list(set(x))).values
del op_s; gc.collect()

te = TransactionEncoder()
te_arr = te.fit_transform(baskets)
df_b = pd.DataFrame(te_arr, columns=te.columns_)
del te_arr, baskets; gc.collect()
print(f"Basket matrix: {df_b.shape}")

freq_a = apriori(df_b, min_support=MIN_SUPPORT, use_colnames=True, max_len=2)
del df_b; gc.collect()
print(f"Frequent itemsets: {len(freq_a)}")

rules_a = association_rules(freq_a, metric='confidence', min_threshold=MIN_CONFIDENCE)
rules_a = rules_a[rules_a.lift >= MIN_LIFT].sort_values('lift', ascending=False)
rules_a['antecedents_str'] = rules_a['antecedents'].apply(lambda x: ', '.join(sorted(x)))
rules_a['consequents_str'] = rules_a['consequents'].apply(lambda x: ', '.join(sorted(x)))
rules_a['rule'] = rules_a['antecedents_str'] + ' → ' + rules_a['consequents_str']
print(f"Rules generated: {len(rules_a)} | Lift range: "
      f"{rules_a.lift.min():.2f}–{rules_a.lift.max():.2f}")

print("\nTop 10 Aisle Rules:")
for _, r in rules_a.head(10).iterrows():
    print(f"  {r['rule']}  | sup={r['support']:.4f} conf={r['confidence']:.2f} lift={r['lift']:.2f}")

# ═══════════════════════════════════════════════════════════════════════
# PRODUCT-LEVEL RULES  (top 60 products, 30K orders)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Product-Level Association Rules ──")
top60 = op.product_name.value_counts().head(60).index.tolist()
op_p  = op[op.product_name.isin(top60)]
s_oids = op_p.order_id.drop_duplicates().sample(min(30_000, op_p.order_id.nunique()), random_state=42)
op_p  = op_p[op_p.order_id.isin(s_oids)]
baskets2 = [b for b in op_p.groupby('order_id')['product_name']
            .apply(lambda x: list(set(x))).values if len(b) >= 2]
del op_p; gc.collect()

te2 = TransactionEncoder()
te_arr2 = te2.fit_transform(baskets2)
df_b2 = pd.DataFrame(te_arr2, columns=te2.columns_)
del te_arr2, baskets2; gc.collect()

freq_p = apriori(df_b2, min_support=0.008, use_colnames=True, max_len=2)
del df_b2; gc.collect()

rules_p = pd.DataFrame()
if len(freq_p) > 0:
    rules_p = association_rules(freq_p, metric='confidence', min_threshold=0.1)
    rules_p = rules_p[rules_p.lift >= 1.0].sort_values('lift', ascending=False)
    rules_p['antecedents_str'] = rules_p['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules_p['consequents_str'] = rules_p['consequents'].apply(lambda x: ', '.join(sorted(x)))
    rules_p['rule'] = rules_p['antecedents_str'] + ' → ' + rules_p['consequents_str']
print(f"Product rules: {len(rules_p)}")

# ─── VISUALIZATIONS ───────────────────────────────────────────────────
print("\nGenerating visualizations...")

if len(rules_a) > 0:
    # Fig 11: Top Rules by Lift
    fig, ax = plt.subplots(figsize=(12, 9))
    top20 = rules_a.head(20)
    bars = ax.barh(range(len(top20)), top20['lift'].values, color=PALETTE[0], alpha=0.85)
    ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20['rule'].values, fontsize=9)
    ax.invert_yaxis(); ax.set_title('Top 20 Association Rules by Lift (Aisle-Level)')
    ax.set_xlabel('Lift')
    for bar, conf in zip(bars, top20['confidence'].values):
        ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                f'conf={conf:.0%}', ha='left', va='center', fontsize=8)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='Lift = 1 (no association)')
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/11_top_rules_lift.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

    # Fig 12: Support vs Confidence
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sizes = np.clip(rules_a['lift'].values * 20, 10, 500)
    sc = ax.scatter(rules_a['support'], rules_a['confidence'],
                    c=rules_a['lift'], cmap='YlOrRd', alpha=0.6,
                    s=sizes, edgecolors='gray', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label='Lift')
    ax.set_title('Association Rules: Support vs Confidence')
    ax.set_xlabel('Support'); ax.set_ylabel('Confidence')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/12_support_vs_confidence.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

if len(rules_p) > 0:
    # Fig 13: Product Rules
    fig, ax = plt.subplots(figsize=(12, 9))
    top20p = rules_p.head(20)
    bars = ax.barh(range(len(top20p)), top20p['lift'].values, color=PALETTE[2], alpha=0.85)
    ax.set_yticks(range(len(top20p))); ax.set_yticklabels(top20p['rule'].values, fontsize=8)
    ax.invert_yaxis(); ax.set_title('Top 20 Product Association Rules by Lift')
    ax.set_xlabel('Lift')
    for bar, conf in zip(bars, top20p['confidence'].values):
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                f'conf={conf:.0%}', ha='left', va='center', fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/13_product_rules.png", dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

# Fig 14: Aisle co-occurrence heatmap
print("Building co-occurrence heatmap...")
top12 = op.aisle.value_counts().head(12).index.tolist()
op_ta = op[op.aisle.isin(top12)][['order_id','aisle']].drop_duplicates()
s_oids2 = op_ta.order_id.drop_duplicates().sample(min(50_000, op_ta.order_id.nunique()), random_state=42)
op_ta = op_ta[op_ta.order_id.isin(s_oids2)]
pairs = op_ta.merge(op_ta, on='order_id', suffixes=('_a','_b'))
pairs = pairs[pairs.aisle_a < pairs.aisle_b]
cc = pairs.groupby(['aisle_a','aisle_b']).size().reset_index(name='count')
del pairs, op_ta; gc.collect()
cooccur = pd.DataFrame(0, index=top12, columns=top12)
for _, row in cc.iterrows():
    cooccur.loc[row['aisle_a'], row['aisle_b']] = row['count']
    cooccur.loc[row['aisle_b'], row['aisle_a']] = row['count']

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(cooccur, dtype=bool), k=1)
sns.heatmap(cooccur, mask=mask, cmap='YlOrRd', annot=True, fmt=',', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Co-occurrence Count'})
ax.set_title('Aisle Co-occurrence Heatmap (Top 12 Aisles)')
plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/14_aisle_cooccurrence.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ─── SAVE ─────────────────────────────────────────────────────────────
if len(rules_a) > 0:
    rules_a[['antecedents_str','consequents_str','support','confidence','lift']].head(TOP_N_RULES) \
        .to_csv(f"{DATA_PROCESSED}/association_rules_aisle.csv", index=False)
if len(rules_p) > 0:
    rules_p[['antecedents_str','consequents_str','support','confidence','lift']].head(TOP_N_RULES) \
        .to_csv(f"{DATA_PROCESSED}/association_rules_product.csv", index=False)

ar_summary = {
    "aisle_rules_count":       int(len(rules_a)),
    "product_rules_count":     int(len(rules_p)),
    "avg_lift_aisle":          float(rules_a['lift'].mean()) if len(rules_a) > 0 else 0,
    "max_lift_aisle":          float(rules_a['lift'].max()) if len(rules_a) > 0 else 0,
    "top_rule":                rules_a.iloc[0]['rule'] if len(rules_a) > 0 else "N/A",
    "top_rule_lift":           float(rules_a.iloc[0]['lift']) if len(rules_a) > 0 else 0,
    "frequent_itemsets_count": int(len(freq_a)),
}
with open(f"{DATA_PROCESSED}/association_rules_summary.json", 'w') as f:
    json.dump(ar_summary, f, indent=2)

print(f"\n✓ Association rules complete! Time: {time.time()-t0:.1f}s")
