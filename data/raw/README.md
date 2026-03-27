# Raw Data

This folder holds the Instacart Market Basket dataset. The CSV files are excluded from version control (`.gitignore`) because of their size (the full dataset is ~750 MB).

## Download Instructions

### Option 1 — Kaggle (recommended, uses real data)

1. Create a free account at [kaggle.com](https://kaggle.com) if you don't have one.
2. Accept the dataset terms at: [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)
3. Download and unzip all CSV files into this folder.

**Using the Kaggle CLI:**
```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/kaggle.json
kaggle competitions download -c instacart-market-basket-analysis
unzip instacart-market-basket-analysis.zip -d data/raw/
```

After downloading, your folder should look like this:

```
data/raw/
├── orders.csv                    # 3.4M rows — user order history
├── order_products__prior.csv     # 32.4M rows — historical purchases (main file)
├── order_products__train.csv     # 1.4M rows — Kaggle competition labels
├── products.csv                  # 49,688 products with aisle & department IDs
├── aisles.csv                    # 134 aisles
└── departments.csv               # 21 departments
```

### Option 2 — Synthetic data (no download needed)

If you just want to explore the code without the full dataset, run the synthetic data generator from the project root:

```bash
python generate_data.py
```

This creates ~200K rows of realistic synthetic data matching the exact schema, so all 4 analysis scripts run end-to-end without modification.

## Dataset Credit

Instacart (2017). *Instacart Market Basket Analysis* [Dataset]. Retrieved from Kaggle:
https://www.kaggle.com/competitions/instacart-market-basket-analysis
