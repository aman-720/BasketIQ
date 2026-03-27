"""
BasketIQ - Module 1: Exploratory Data Analysis
Profiles the dataset, generates summary statistics, and produces EDA visualizations.
Memory-optimised for the full 32M-row Instacart dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, json, warnings, gc
warnings.filterwarnings('ignore')

from config import *

# ─── STYLE SETUP ──────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': FIG_DPI, 'font.size': 11, 'axes.titlesize': 14,
    'axes.titleweight': 'bold', 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────
# Use optimized dtypes to keep 32M+ row file memory-efficient
print("Loading data...")
DTYPES_OP = {'order_id': np.int32, 'product_id': np.int32,
             'add_to_cart_order': np.int16, 'reordered': np.int8}
DTYPES_ORD = {'order_id': np.int32, 'user_id': np.int32,
              'order_number': np.int16, 'order_dow': np.int8,
              'order_hour_of_day': np.int8}

orders = pd.read_csv(f"{DATA_RAW}/orders.csv", dtype=DTYPES_ORD)
order_products = pd.read_csv(f"{DATA_RAW}/order_products__prior.csv",
                              usecols=['order_id','product_id','reordered'],
                              dtype=DTYPES_OP)
products = pd.read_csv(f"{DATA_RAW}/products.csv",
                        usecols=['product_id','product_name','aisle_id','department_id'])
aisles       = pd.read_csv(f"{DATA_RAW}/aisles.csv")
departments  = pd.read_csv(f"{DATA_RAW}/departments.csv")

# Build lookup dictionaries — small, fast, memory-cheap
products_full = products.merge(aisles, on='aisle_id').merge(departments, on='department_id')
pid_to_name   = dict(zip(products_full.product_id, products_full.product_name))
pid_to_aisle  = dict(zip(products_full.product_id, products_full.aisle))
pid_to_dept   = dict(zip(products_full.product_id, products_full.department))
del products, aisles, departments; gc.collect()

orders_prior = orders[orders.eval_set == 'prior'].copy()

print(f"Loaded: {len(order_products):,} transactions | "
      f"{orders.user_id.nunique():,} users | "
      f"{len(products_full):,} products | "
      f"{orders_prior.order_id.nunique():,} orders")

# ─── SUMMARY STATS ────────────────────────────────────────────────────
print("\n── Summary Statistics ──")
items_per_order = order_products.groupby('order_id').size()
summary = {
    "total_transactions":    int(len(order_products)),
    "total_orders":          int(orders_prior.order_id.nunique()),
    "total_users":           int(orders.user_id.nunique()),
    "total_products":        int(len(products_full)),
    "total_departments":     int(len(set(pid_to_dept.values()))),
    "total_aisles":          int(len(set(pid_to_aisle.values()))),
    "avg_items_per_order":   float(items_per_order.mean()),
    "median_items_per_order":float(items_per_order.median()),
    "avg_orders_per_user":   float(orders_prior.groupby('user_id').size().mean()),
    "reorder_rate":          float(order_products.reordered.mean()),
    "avg_days_between_orders":float(orders.days_since_prior_order.mean()),
}
with open(f"{DATA_PROCESSED}/eda_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
for k, v in summary.items():
    print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v:,}")

# ─── FIGURE 1: Orders by Day of Week ─────────────────────────────────
fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
dow_labels = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
dow_counts = orders_prior.order_dow.value_counts().sort_index()
bars = ax.bar(range(7), dow_counts.values, color=PALETTE[0], edgecolor='white', width=0.7)
ax.set_xticks(range(7))
ax.set_xticklabels(dow_labels, rotation=30, ha='right')
ax.set_title('Orders by Day of Week')
ax.set_ylabel('Number of Orders')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
            f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/01_orders_by_dow.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ─── FIGURE 2: Orders by Hour of Day ─────────────────────────────────
fig, ax = plt.subplots(figsize=FIG_SIZE)
hour_counts = orders_prior.order_hour_of_day.value_counts().sort_index()
ax.fill_between(hour_counts.index, hour_counts.values, alpha=0.3, color=PALETTE[0])
ax.plot(hour_counts.index, hour_counts.values, color=PALETTE[0], linewidth=2.5,
        marker='o', markersize=5)
ax.set_title('Orders by Hour of Day')
ax.set_xlabel('Hour (24h)'); ax.set_ylabel('Number of Orders')
ax.set_xticks(range(0, 24))
peak = hour_counts.idxmax()
ax.annotate(f'Peak: {peak}:00\n({hour_counts.max():,})',
            xy=(peak, hour_counts.max()), xytext=(peak+2, hour_counts.max()*0.93),
            arrowprops=dict(arrowstyle='->', color=PALETTE[3]), fontsize=10, color=PALETTE[3])
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/02_orders_by_hour.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ─── FIGURE 3: Top 20 Products ─── (use map, no big merge)
print("Fig 3: Top products...")
top_prods = order_products.product_id.map(pid_to_name).value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(len(top_prods)), top_prods.values, color=PALETTE[2])
ax.set_yticks(range(len(top_prods))); ax.set_yticklabels(top_prods.index); ax.invert_yaxis()
ax.set_title('Top 20 Most Ordered Products')
ax.set_xlabel('Number of Orders')
for bar in bars:
    ax.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():,.0f}', ha='left', va='center', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/03_top20_products.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ─── FIGURE 4: Department Distribution ───────────────────────────────
print("Fig 4: Department distribution...")
dept_counts = order_products.product_id.map(pid_to_dept).value_counts().head(15)
colors = [PALETTE[i % len(PALETTE)] for i in range(len(dept_counts))]
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(len(dept_counts)), dept_counts.values, color=colors)
ax.set_yticks(range(len(dept_counts)))
ax.set_yticklabels(dept_counts.index.str.title()); ax.invert_yaxis()
ax.set_title('Orders by Department (Top 15)')
ax.set_xlabel('Number of Items Ordered')
for bar in bars:
    ax.text(bar.get_width()+1000, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():,.0f}', ha='left', va='center', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/04_department_distribution.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ─── FIGURE 5: Items per Order Distribution ──────────────────────────
fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
ax.hist(items_per_order.clip(upper=60).values, bins=60, color=PALETTE[0],
        edgecolor='white', alpha=0.85)
ax.axvline(items_per_order.mean(), color=PALETTE[3], linestyle='--', linewidth=2,
           label=f'Mean: {items_per_order.mean():.1f}')
ax.axvline(items_per_order.median(), color=PALETTE[2], linestyle='--', linewidth=2,
           label=f'Median: {items_per_order.median():.1f}')
ax.set_title('Distribution of Items per Order')
ax.set_xlabel('Number of Items'); ax.set_ylabel('Number of Orders')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/05_items_per_order.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()
del items_per_order; gc.collect()

# ─── FIGURE 6: Reorder Rate by Department ────────────────────────────
print("Fig 6: Reorder by department...")
order_products['department'] = order_products.product_id.map(pid_to_dept)
reorder_dept = order_products.groupby('department')['reordered'].mean().sort_values().tail(15)
order_products.drop(columns=['department'], inplace=True); gc.collect()
colors = [PALETTE[2] if v > 0.6 else PALETTE[0] if v > 0.5 else PALETTE[3]
          for v in reorder_dept.values]
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(reorder_dept)), reorder_dept.values, color=colors)
ax.set_yticks(range(len(reorder_dept)))
ax.set_yticklabels(reorder_dept.index.str.title())
ax.set_title('Reorder Rate by Department')
ax.set_xlabel('Reorder Rate')
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
for bar in bars:
    ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():.1%}', ha='left', va='center', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/06_reorder_rate_by_dept.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ─── FIGURE 7: Days Since Prior Order ────────────────────────────────
fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
days = orders.days_since_prior_order.dropna()
ax.hist(days, bins=30, color=PALETTE[4], edgecolor='white', alpha=0.85)
ax.axvline(days.mean(), color=PALETTE[3], linestyle='--', linewidth=2,
           label=f'Mean: {days.mean():.1f} days')
ax.set_title('Days Between Consecutive Orders')
ax.set_xlabel('Days'); ax.set_ylabel('Number of Orders')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/07_days_between_orders.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()
del days; gc.collect()

# ─── FIGURE 8: Heatmap — Orders by Day & Hour ────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
pivot = orders_prior.groupby(['order_dow','order_hour_of_day']).size().unstack(fill_value=0)
pivot.index = dow_labels
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3,
            cbar_kws={'label': 'Number of Orders'})
ax.set_title('Order Heatmap: Day of Week vs Hour of Day')
ax.set_xlabel('Hour of Day'); ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/08_order_heatmap.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()
del pivot; gc.collect()

# ─── FIGURE 9: Top 15 Aisles ─────────────────────────────────────────
print("Fig 9: Top aisles...")
aisle_counts = order_products.product_id.map(pid_to_aisle).value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(aisle_counts)), aisle_counts.values, color=PALETTE[1])
ax.set_yticks(range(len(aisle_counts))); ax.set_yticklabels(aisle_counts.index.str.title())
ax.invert_yaxis(); ax.set_title('Top 15 Aisles by Items Ordered')
ax.set_xlabel('Number of Items')
for bar in bars:
    ax.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():,.0f}', ha='left', va='center', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/09_top15_aisles.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()

# ─── FIGURE 10: Orders per User Distribution ─────────────────────────
fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
opu = orders_prior.groupby('user_id').size()
ax.hist(opu.clip(upper=100).values, bins=50, color=PALETTE[5], edgecolor='white', alpha=0.85)
ax.axvline(opu.mean(), color=PALETTE[3], linestyle='--', linewidth=2,
           label=f'Mean: {opu.mean():.1f} orders')
ax.set_title('Distribution of Orders per User')
ax.set_xlabel('Number of Orders'); ax.set_ylabel('Number of Users')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/10_orders_per_user.png", dpi=FIG_DPI, bbox_inches='tight')
plt.close()
del opu; gc.collect()

# ─── SAVE PROCESSED DATA ─────────────────────────────────────────────
# Top products
top_prods.reset_index().rename(columns={'index':'product_name','product_id':'order_count'}) \
    .to_csv(f"{DATA_PROCESSED}/top_products.csv", index=False)

# Department summary — computed directly from dictionaries
order_products['department'] = order_products.product_id.map(pid_to_dept)
order_products['aisle'] = order_products.product_id.map(pid_to_aisle)

dept_summary = order_products.groupby('department').agg(
    total_items=('product_id','count'),
    unique_products=('product_id','nunique'),
    reorder_rate=('reordered','mean')
).sort_values('total_items', ascending=False).reset_index()
dept_summary.to_csv(f"{DATA_PROCESSED}/department_summary.csv", index=False)

aisle_summary = order_products.groupby('aisle').agg(
    total_items=('product_id','count'),
    unique_products=('product_id','nunique'),
    reorder_rate=('reordered','mean')
).sort_values('total_items', ascending=False).reset_index()
aisle_summary.to_csv(f"{DATA_PROCESSED}/aisle_summary.csv", index=False)

orders_prior.order_hour_of_day.value_counts().sort_index().reset_index() \
    .rename(columns={'order_hour_of_day':'hour','count':'count'}) \
    .to_csv(f"{DATA_PROCESSED}/orders_by_hour.csv", index=False)

dow_df = orders_prior.order_dow.value_counts().sort_index().reset_index()
dow_df.columns = ['dow','count']
dow_df['day_name'] = dow_labels
dow_df.to_csv(f"{DATA_PROCESSED}/orders_by_dow.csv", index=False)

print(f"\n✓ 10 EDA visualizations → {VIZ_DIR}/")
print(f"✓ Processed data       → {DATA_PROCESSED}/")
print("EDA complete.")
