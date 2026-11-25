"""BANA 212 Group Project - Finding Affordable Alternatives to Luxury Skincare Products.

Streamlit app that presents the analysis. All models are trained in the notebook and loaded
from models/; nothing is retrained here.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
MODEL_DIR, DATA_DIR = BASE, BASE

st.set_page_config(page_title="Luxury Skincare Alternatives", page_icon="🧴", layout="wide")

COLORS = {"Face Serums": "#2a78b8", "Moisturizers": "#e07b39"}
TIER_ORDER = ["Affordable", "Mid-range", "Expensive"]


# ---------- loaders (cached so files are read once per session) ----------
@st.cache_data
def load_data():
    products = pd.read_csv(DATA_DIR / "products_clean.csv")
    alternatives = pd.read_csv(DATA_DIR / "alternatives.csv")
    return products, alternatives


@st.cache_data
def load_results():
    with open(MODEL_DIR / "model_results.json") as f:
        return json.load(f)


@st.cache_resource
def load_model(target, model_name):
    fname = model_name.lower().replace(" ", "_").replace("-", "_")
    return joblib.load(MODEL_DIR / f"{target}__{fname}.joblib")


products, alternatives = load_data()
R = load_results()
results = pd.DataFrame(R["results"])
TARGET_LABELS = {t: v["label"] for t, v in R["targets"].items()}
MODEL_NAMES = list(results["model"].unique())

st.title("Finding Affordable Alternatives to Luxury Skincare")
st.caption("BANA 212 Group Project · Sephora face serums and moisturizers (Kaggle: Sephora Products and Skincare Reviews)")

tabs = st.tabs(["Executive Summary", "Descriptive Analytics", "Alternative Finder",
                "Model Performance", "Interactive Prediction"])

# =====================================================================
# 1. EXECUTIVE SUMMARY
# =====================================================================
with tabs[0]:
    n_exp = (products["price_tier"] == "Expensive").sum()
    n_cov = len(alternatives[["brand_name_exp", "product_name_exp"]].drop_duplicates())
    best = results.loc[results.groupby("target")["test_auc"].idxmax()].set_index("target")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products analyzed", f"{len(products):,}")
    c2.metric("Price–rating correlation", "0.10 – 0.13", help="Spearman, serums and moisturizers")
    c3.metric("Luxury products with a candidate", f"{n_cov} of {n_exp}")
    c4.metric("Median savings", "67% – 74%", help="Serums 67%, moisturizers 74%")

    st.subheader("The data and the question")
    st.write(
        f"We studied {len(products):,} face serums and moisturizers sold at Sephora, using a public dataset "
        "that lists each product's price, size, full ingredient list, average customer rating, and number of "
        "reviews. After removing mini sizes, duplicates, and products without a usable ingredient list, we split "
        "each category into three equal price groups: Affordable, Mid-range, and Expensive. Our central question "
        "is whether shoppers can find lower-priced products that are similar to expensive ones in what they "
        "contain and in how customers rate them. For the predictive part, we built two targets: whether a "
        "product falls in the Expensive group, and whether it earns a high rating of 4.5 stars or more."
    )
    st.subheader("Why it matters")
    st.write(
        "Prestige skincare often costs $80 to $400 per bottle, and shoppers rarely know whether the price "
        "reflects the formula or the brand. If affordable products share many of the same key ingredients and "
        "are rated just as well, consumers can save a large share of their spending without giving up quality. "
        "For a retailer, the same logic powers a \"you may also like\" feature that keeps price-sensitive "
        "customers in the store instead of losing them to another site."
    )
    st.subheader("Our approach and what we found")
    st.write(
        "First, we compared prices with ratings and found almost no link: expensive products are not rated "
        "meaningfully higher, and affordable ones often have more reviews behind their scores. Second, we "
        "compared every expensive product with every affordable product in the same category from a different "
        "brand, measuring how many key ingredients they share after removing basic ingredients like water and "
        "preservatives. Third, we kept only pairs where the cheaper product shares enough key ingredients, is "
        f"rated at least 4.0 and close to the luxury product, and has at least 50 reviews. This produced "
        f"{len(alternatives)} recommended pairs covering {n_cov} expensive products, with typical savings of "
        "about two-thirds of the price."
    )
    st.write(
        "Finally, we trained two machine learning models (Logistic Regression and Random Forest) for each target. Using only formula and listing "
        f"details (not brand), the best model identifies Expensive products with a ROC AUC of "
        f"{best.loc['is_expensive', 'test_auc']:.2f} and high-rated products with "
        f"{best.loc['is_high_rated', 'test_auc']:.2f}, where 0.5 is a coin flip and 1.0 is perfect. Because "
        "the models catch fewer than half of the expensive products, many luxury items are hard to tell apart "
        "from cheaper ones on everything except price and brand, which supports the case for trying an "
        "affordable alternative first."
    )
    st.info("Candidates are possible alternatives, not proven equivalents: ingredient lists do not show "
            "concentrations, and the data is a historical snapshot of Sephora's catalog.")

# =====================================================================
# 2. DESCRIPTIVE ANALYTICS
# =====================================================================
with tabs[1]:
    st.subheader("1. Price distribution by category")
    fig = px.box(products, x="category", y="price_usd", color="category", points="outliers",
                 color_discrete_map=COLORS, labels={"price_usd": "Price (USD)", "category": ""},
                 hover_data=["brand_name", "product_name"])
    fig.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Serums and moisturizers have similar typical prices (median $65 vs. $60), but both have a long tail of "
        "luxury products reaching $400 or more, such as La Mer and SK-II. Because of this skew, we use medians, "
        "rank-based correlations, and price groups defined within each category rather than raw averages."
    )

    st.subheader("2. Customer rating distribution")
    fig = px.histogram(products, x="rating", color="category", nbins=25, barmode="overlay", opacity=0.65,
                       color_discrete_map=COLORS, labels={"rating": "Average rating (1–5)", "count": "Products"})
    fig.update_layout(height=400, yaxis_title="Products")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Ratings are concentrated between about 4.1 and 4.6 stars in both categories (mean 4.30 for serums and "
        "4.27 for moisturizers). Since almost every product is rated well, a good rating alone cannot identify an "
        "alternative, which is why we also use ingredient overlap and review counts."
    )

    st.subheader("3. Does a higher price mean a higher rating?")
    cat = st.radio("Category", ["Both", "Face Serums", "Moisturizers"], horizontal=True, key="scatter_cat")
    sub = products if cat == "Both" else products[products["category"] == cat]
    fig = px.scatter(sub, x="price_usd", y="rating", color="category", size="reviews", size_max=28,
                     opacity=0.55, color_discrete_map=COLORS, hover_data=["brand_name", "product_name", "reviews"],
                     labels={"price_usd": "Price (USD)", "rating": "Average rating"})
    fig.update_layout(height=460)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each bubble is a product, sized by number of reviews. The cloud is nearly flat: the Spearman correlation "
        "between price and rating is only 0.10 for serums and 0.13 for moisturizers, so paying more buys "
        "almost no improvement in customer rating."
    )

    st.subheader("4. Rating by price tier")
    tier = (products.groupby(["category", "price_tier"]).agg(mean_rating=("rating", "mean"),
                                                             median_reviews=("reviews", "median"))
            .reset_index())
    fig = px.bar(tier, x="price_tier", y="mean_rating", color="category", barmode="group",
                 category_orders={"price_tier": TIER_ORDER}, color_discrete_map=COLORS, text_auto=".2f",
                 hover_data=["median_reviews"], labels={"price_tier": "", "mean_rating": "Average rating"})
    fig.update_layout(height=400, yaxis_range=[3.8, 4.5])
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Affordable serums average 4.25 stars against 4.28 for expensive ones, and mid-range serums are rated "
        "highest of all. Moisturizers rise only slightly with price (4.19 to 4.33), while affordable products "
        "carry more reviews, so their ratings rest on more customer feedback."
    )

    st.subheader("5. Alternative candidates: similarity vs. savings")
    fig = px.scatter(alternatives, x="jaccard", y="savings_pct", color="category", size="reviews_aff",
                     size_max=25, opacity=0.65, color_discrete_map=COLORS,
                     hover_data=["brand_name_exp", "product_name_exp", "brand_name_aff", "product_name_aff",
                                 "price_usd_exp", "price_usd_aff"],
                     labels={"jaccard": "Ingredient similarity (Jaccard, key ingredients)",
                             "savings_pct": "Savings vs. expensive product"})
    fig.update_layout(height=460, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Each point is one of the {len(alternatives)} expensive–affordable pairs that pass all screening rules. "
        "Most candidates save 50–90% of the price while sharing at least five key ingredients; pairs toward the "
        "upper right are both the most similar and the cheapest, making them the strongest recommendations."
    )

# =====================================================================
# 3. ALTERNATIVE FINDER
# =====================================================================
with tabs[2]:
    st.subheader("Look up affordable alternatives for a luxury product")
    st.write("Pick an expensive product to see the lower-priced candidates that passed our screening rules "
             "(ingredient similarity ≥ 0.15, ≥ 5 shared key ingredients, rating ≥ 4.0 and within 0.2 stars, "
             "≥ 50 reviews).")
    lux = (alternatives.assign(label=lambda d: d["brand_name_exp"] + " – " + d["product_name_exp"]
                               + " ($" + d["price_usd_exp"].round(0).astype(int).astype(str) + ")")
           ["label"].drop_duplicates().sort_values().tolist())
    default = next((i for i, l in enumerate(lux) if "Protini" in l), 0)
    choice = st.selectbox(f"Luxury product ({len(lux)} with at least one candidate)", lux, index=default)
    rows = alternatives[(alternatives["brand_name_exp"] + " – " + alternatives["product_name_exp"]
                         + " ($" + alternatives["price_usd_exp"].round(0).astype(int).astype(str) + ")") == choice]
    rows = rows.sort_values(["jaccard", "rating_aff", "savings_pct"], ascending=False)
    lux_row = rows.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Luxury price", f"${lux_row['price_usd_exp']:.0f}")
    m2.metric("Luxury rating", f"{lux_row['rating_exp']:.2f} ★")
    m3.metric("Best saving", f"{rows['savings_pct'].max():.0%}")
    show = rows[["brand_name_aff", "product_name_aff", "price_usd_aff", "rating_aff", "reviews_aff",
                 "n_shared", "jaccard", "savings_pct"]].rename(columns={
        "brand_name_aff": "Brand", "product_name_aff": "Product", "price_usd_aff": "Price ($)",
        "rating_aff": "Rating", "reviews_aff": "Reviews", "n_shared": "Shared key ingredients",
        "jaccard": "Similarity", "savings_pct": "Savings"})
    st.dataframe(show.style.format({"Price ($)": "{:.2f}", "Rating": "{:.2f}", "Reviews": "{:,.0f}",
                                    "Similarity": "{:.2f}", "Savings": "{:.0%}"}),
                 hide_index=True, width="stretch")
    with st.expander("Shared key ingredients with the top candidate"):
        st.write(rows.iloc[0]["shared_ingredients"].replace("; ", ", "))

# =====================================================================
# 4. MODEL PERFORMANCE
# =====================================================================
with tabs[3]:
    target = st.radio("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, horizontal=True,
                      key="perf_target")
    res = results[results["target"] == target].drop(columns="target").set_index("model")
    pos = R["targets"][target]["positive_rate"]
    st.write(f"Test set: {R['n_test']} products (20%, stratified). Positive class share: {pos:.0%}, so always "
             f"predicting the majority class gives {1 - pos:.0%} accuracy. Models were tuned with 5-fold "
             "cross-validated grid search on ROC AUC.")

    st.subheader("Comparison table")
    table = res.rename(columns={"cv_auc": "CV AUC", "test_auc": "Test AUC", "accuracy": "Accuracy",
                                "precision": "Precision", "recall": "Recall", "f1": "F1"})
    st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8d6"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Metric comparison")
        long = table[["Test AUC", "Accuracy", "Precision", "Recall", "F1"]].reset_index().melt(
            id_vars="model", var_name="metric", value_name="score")
        fig = px.bar(long, x="metric", y="score", color="model", barmode="group", text_auto=".2f",
                     labels={"metric": "", "score": "Score"})
        fig.update_layout(height=430, yaxis_range=[0, 1], legend_title="")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("ROC curves (test set)")
        fig = go.Figure()
        for name, pts in R["roc"][target].items():
            fig.add_trace(go.Scatter(x=pts["fpr"], y=pts["tpr"], mode="lines",
                                     name=f"{name} (AUC {res.loc[name, 'test_auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(dash="dash", color="grey")))
        fig.update_layout(height=430, xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(x=0.45, y=0.05))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Best hyperparameters")
    hp = pd.DataFrame([{"Model": m, "Best hyperparameters": ", ".join(f"{k} = {v}" for k, v in p.items())}
                       for m, p in R["best_params"][target].items()])
    st.dataframe(hp, hide_index=True, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Confusion matrix")
        cm_model = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(res["test_auc"].idxmax()),
                                key="cm_model")
        cm = np.array(R["confusion"][target][cm_model])
        lbl = TARGET_LABELS[target]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=[f"Pred: not {lbl}", f"Pred: {lbl}"], y=[f"Actual: not {lbl}", f"Actual: {lbl}"])
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with c4:
        imp = R["importance"][target]
        st.subheader(f"Top drivers ({imp['model']})")
        s = pd.Series(imp["importance"]).sort_values(ascending=False).head(10)[::-1]
        fig = px.bar(x=s.values, y=s.index, orientation="h", labels={"x": "Drop in test AUC when shuffled", "y": ""})
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

    if target == "is_expensive":
        st.success(
            "Random Forest performs best (test AUC 0.73, CV AUC 0.83) but catches fewer than half of the "
            "expensive products. Fragrance ingredients and longer ingredient lists push toward the Expensive "
            "tier, while Sephora-exclusive and 'Clean at Sephora' products lean cheaper. Formula details explain "
            "part of the price, but much of the luxury premium is invisible in the product's attributes."
        )
    else:
        st.success(
            "High ratings are harder to predict (best test AUC about 0.66). Review and love counts matter most, "
            "and price ranks only third, which echoes the weak price–rating link in the descriptive analysis. "
            "Random Forest also beats Logistic Regression on recall (0.13 vs. 0.04), but both models flag few "
            "top-rated products at the 0.5 threshold."
        )

# =====================================================================
# 5. INTERACTIVE PREDICTION
# =====================================================================
FEATURE_LABELS = {
    "is_serum": "Category", "size_ml": "Size (mL)", "n_ingredients": "Number of ingredients",
    "log_reviews": "Number of reviews", "log_loves": "Number of loves (favorites)", "rating": "Average rating",
    "price_usd": "Price (USD)", "has_fragrance": "Contains fragrance", "sephora_exclusive": "Sephora exclusive",
    "is_clean": "Clean at Sephora", "has_peptide": "Contains peptides", "has_retinoid": "Contains retinoid",
}

with tabs[4]:
    st.write("Choose a target and a model, set the key product attributes, and the saved model returns a live "
             "prediction. All other features are held at their dataset averages.")
    c1, c2 = st.columns(2)
    p_target = c1.selectbox("What to predict", list(TARGET_LABELS), format_func=TARGET_LABELS.get, key="pred_target")
    p_model = c2.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("Random Forest"), key="pred_model")
    features = R["targets"][p_target]["features"]
    stats = R["feature_stats"]

    # start every feature at its average
    x = {f: stats[f]["mean"] for f in features}

    st.markdown("##### Product attributes")
    a, b, c = st.columns(3)
    with a:
        x["is_serum"] = 1 if st.radio("Category", ["Face Serums", "Moisturizers"], horizontal=True) == "Face Serums" else 0
        x["size_ml"] = st.slider("Size (mL)", 5, 200, int(stats["size_ml"]["median"]), step=5)
        x["n_ingredients"] = st.slider("Number of ingredients", 5, 100, int(round(stats["n_ingredients"]["mean"])))
    with b:
        reviews = st.number_input("Number of reviews", 0, 20000, int(np.expm1(stats["log_reviews"]["median"])), step=50)
        loves = st.number_input("Number of loves (favorites)", 0, 1_000_000, int(np.expm1(stats["log_loves"]["median"])),
                                step=1000)
        x["log_reviews"], x["log_loves"] = np.log1p(reviews), np.log1p(loves)
        if p_target == "is_expensive":
            x["rating"] = st.slider("Average rating", 1.0, 5.0, round(stats["rating"]["mean"], 2), step=0.05)
        else:
            x["price_usd"] = st.slider("Price (USD)", 5, 425, int(round(stats["price_usd"]["median"])))
    with c:
        for f in ["has_fragrance", "has_peptide", "sephora_exclusive", "is_clean"]:
            x[f] = int(st.toggle(FEATURE_LABELS[f], value=stats[f]["mean"] >= 0.5))

    X = pd.DataFrame([x])[features]
    prob = float(load_model(p_target, p_model).predict_proba(X)[0, 1])
    label = TARGET_LABELS[p_target]

    st.markdown("##### Prediction")
    g1, g2 = st.columns([1, 2])
    with g1:
        st.metric(f"Probability: {label}", f"{prob:.0%}")
        if prob >= 0.5:
            st.success(f"Predicted: **{label}**")
        else:
            st.warning(f"Predicted: **not {label}**")
        st.caption(f"Base rate in the data: {R['targets'][p_target]['positive_rate']:.0%}")
    with g2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2a78b8"},
                   "threshold": {"line": {"color": "red", "width": 3}, "value": 50}}))
        fig.update_layout(height=250, margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")

    with st.expander("All feature values sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}).round(3), width="stretch")
