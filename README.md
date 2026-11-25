# Finding Affordable Alternatives to Luxury Skincare Products

**BANA 212 Group Project · UC Irvine MSBA**

🔗 **Live app:** _paste your Streamlit URL here_

## Question
Do lower-priced face serums and moisturizers at Sephora share key ingredients with, and get ratings as good as, expensive ones? And can a product's formula and listing details predict whether it is (a) in the Expensive price tier or (b) highly rated (4.5+ stars)?

## Data
[Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) (Kaggle), `product_info.csv`. After cleaning: 702 products (347 face serums, 355 moisturizers).

## Approach
1. **Q1 – Price vs. rating:** distributions, Spearman correlations, ratings by price tier (category-specific terciles).
2. **Q2 – Ingredient overlap:** Jaccard similarity on key ingredients (base ingredients such as water and preservatives removed) for every cross-brand Expensive–Affordable pair.
3. **Q3 – Alternative screening:** Jaccard ≥ 0.15, ≥ 5 shared key ingredients, rating ≥ 4.0 and within 0.2 of the luxury product, ≥ 50 reviews → 103 pairs covering 65 luxury products.
4. **Q4 – Predictive modeling:** Logistic Regression and Random Forest for two binary targets, tuned with 5-fold GridSearchCV on ROC AUC. Brand is excluded; the price model never sees price and the rating model never sees rating.

## Key results
| | Finding |
| :--- | :--- |
| Price ↔ rating | Spearman 0.10 (serums), 0.13 (moisturizers): price barely relates to rating |
| Alternatives | 65 of 232 luxury products have a candidate; median savings 67% (serums), 74% (moisturizers) |
| Expensive tier model | Best: Random Forest, test AUC 0.73 |
| High-rating model | Best: Random Forest, test AUC 0.66 |

## Repository
```
├── app.py                                 # Streamlit app (loads saved models, no retraining)
├── <your notebook>.ipynb                  # Full analysis + model training/export
├── models/                                # 4 tuned sklearn pipelines (.joblib) + model_results.json
├── data/                                  # products_clean.csv, alternatives.csv (exported by notebook)
├── requirements.txt
└── .streamlit/config.toml
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproduce the models
Run the notebook top to bottom (Colab works; it downloads the data with `kagglehub`). Section 4 writes `models/` and `data/`. The saved models in this repo were trained with scikit-learn 1.9.1 on Python 3.13 with the same code and random seed; if you retrain with a different version, update `requirements.txt` to match so the app can load the files.

## Limitations
Ingredient lists show names, not concentrations, so candidates are possible alternatives rather than proven equivalents. Thresholds are our own choices, and the data is a historical snapshot of Sephora's catalog.
