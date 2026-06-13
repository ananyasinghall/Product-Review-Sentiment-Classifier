"""
sentiment_analysis_final.py
============================
Binary sentiment classifier for Amazon Kitchen product reviews.

Pipeline:
    Load → Clean → Preprocess → Vectorise (TF-IDF) → Train → Evaluate → Visualise

Research questions addressed:
    1. Which of four classical ML models best generalises to unseen review text?
    2. Do bigram TF-IDF features improve over unigram-only features?
    3. What vocabulary most strongly drives each sentiment class?
    4. Where do models systematically fail (confusion matrix analysis)?

Dataset:
    https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz
"""

# ── Standard library ────────────────────────────────────────────────────────
import re
import itertools
import warnings

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nltk
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ── NLTK assets ─────────────────────────────────────────────────────────────
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 544
np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Amazon reviews TSV, retain only star_rating and review_body,
    and drop rows with missing values.

    Returns a DataFrame with columns: ['star_rating', 'review_body']
    """
    raw = pd.read_csv(filepath, sep="\t", compression="gzip", on_bad_lines="skip")
    df = raw[["star_rating", "review_body"]].copy()
    df.dropna(inplace=True)
    return df


def describe_raw_distribution(df: pd.DataFrame) -> None:
    """Print per-rating counts so the class imbalance is visible before sampling."""
    total = len(df)
    print(f"Total reviews loaded: {total:,}")
    for rating in range(1, 6):
        count = len(df[df["star_rating"] == rating])
        print(f"  Rating {rating}: {count:>8,}  ({100 * count / total:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# 2. LABELLING AND BALANCING
# ═══════════════════════════════════════════════════════════════════════════

def label_and_balance(df: pd.DataFrame, n_per_class: int = 100_000) -> pd.DataFrame:
    """
    Convert star ratings to binary sentiment labels and sample a balanced corpus.

    Mapping:  4, 5  →  1 (positive)
              1, 2  →  0 (negative)
              3     →  discarded (genuinely ambiguous)

    Args:
        df:           Raw DataFrame with 'star_rating' column.
        n_per_class:  Number of samples to draw from each class.

    Returns:
        Balanced DataFrame with binary 'star_rating' column.
    """
    # Discard neutral reviews
    df = df[df["star_rating"] != 3].copy()

    # Binary label mapping
    df["star_rating"] = df["star_rating"].map({1: 0, 2: 0, 4: 1, 5: 1}).astype("int8")

    # Balanced sample — equal class counts prevent majority-class bias
    positive = df[df["star_rating"] == 1].sample(n_per_class, random_state=RANDOM_SEED)
    negative = df[df["star_rating"] == 0].sample(n_per_class, random_state=RANDOM_SEED)
    balanced = pd.concat([positive, negative]).reset_index(drop=True)

    pos_count = balanced["star_rating"].sum()
    neg_count = len(balanced) - pos_count
    print(f"\nBalanced dataset: {pos_count:,} positive, {neg_count:,} negative")
    return balanced


# ═══════════════════════════════════════════════════════════════════════════
# 3. TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def _remove_html_and_urls(text: str) -> str:
    """Strip HTML tags (via BeautifulSoup) and HTTP/HTTPS URLs."""
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r"http\S+", "", text)
    return text


def _expand_contractions(text: str) -> str:
    """Expand English contractions: won't → will not, etc."""
    return contractions.fix(text)


def _remove_non_alpha(text: str) -> str:
    """Keep only ASCII letters and spaces; replace everything else with a space."""
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def _collapse_whitespace(text: str) -> str:
    """Replace runs of whitespace with a single space and strip ends."""
    return " ".join(text.split())


def clean_text(text: str) -> str:
    """
    Full cleaning pass on a single review string.

    Steps applied in order:
        1. Lowercase
        2. Remove HTML tags and URLs
        3. Expand contractions
        4. Remove non-alphabetical characters
        5. Collapse whitespace
    """
    text = str(text).lower()
    text = _remove_html_and_urls(text)
    text = _expand_contractions(text)
    text = _remove_non_alpha(text)
    text = _collapse_whitespace(text)
    return text


def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to all reviews and store result in 'cleaned_reviews'."""
    df = df.copy()
    df["cleaned_reviews"] = df["review_body"].apply(clean_text)

    before = df["review_body"].apply(lambda x: len(str(x))).mean()
    after  = df["cleaned_reviews"].apply(len).mean()
    print(f"\nCleaning: avg length {before:.0f} → {after:.0f} chars "
          f"({100 * (before - after) / before:.1f}% reduction)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()
_TOKENIZER   = nltk.tokenize.WhitespaceTokenizer()


def preprocess_text(text: str) -> str:
    """
    Preprocessing pass on a single cleaned review string.

    Steps applied in order:
        6. Remove NLTK English stop words
        7. Lemmatize tokens using verb POS (consolidates inflected forms)

    Note: Stop word removal runs before lemmatization intentionally — the
    stop word list uses base forms, so checking after lemmatization would
    still catch them, but this order is marginally faster.
    """
    tokens = [w for w in str(text).split() if w not in _STOP_WORDS]
    tokens = [_LEMMATIZER.lemmatize(w, pos="v") for w in tokens]
    return " ".join(tokens)


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocess_text to all cleaned reviews; store in 'processed_reviews'."""
    df = df.copy()
    df["processed_reviews"] = df["cleaned_reviews"].apply(preprocess_text)

    before = df["cleaned_reviews"].apply(len).mean()
    after  = df["processed_reviews"].apply(len).mean()
    print(f"Preprocessing: avg length {before:.0f} → {after:.0f} chars "
          f"({100 * (before - after) / before:.1f}% reduction)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def build_tfidf_features(review_train, review_test, ngram_range=(1, 1)):
    """
    Fit a TF-IDF vectorizer on training data only and transform both splits.

    Fitting on training data only is critical to prevent data leakage:
    the IDF weights must not be influenced by test-set vocabulary.

    Args:
        review_train:  Training text Series.
        review_test:   Test text Series.
        ngram_range:   Tuple passed to TfidfVectorizer (e.g. (1,1) or (1,2)).

    Returns:
        vectorizer, X_train (sparse), X_test (sparse)
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(review_train)
    X_test  = vectorizer.transform(review_test)
    print(f"\nTF-IDF {ngram_range}: {X_train.shape[1]:,} features "
          f"| train {X_train.shape[0]:,} | test {X_test.shape[0]:,}")
    return vectorizer, X_train, X_test


# ═══════════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred) -> dict:
    """Return a dict of accuracy, precision, recall (macro), and F1."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def report_results(model_name: str, y_train, y_train_pred, y_test, y_test_pred) -> dict:
    """
    Print and return train/test metrics for a fitted model.

    Printing both train and test scores makes overfitting immediately visible:
    a large train-test gap (e.g. train F1=0.99, test F1=0.85) signals that
    the model has memorised the training set rather than generalised.
    """
    train_m = compute_metrics(y_train, y_train_pred)
    test_m  = compute_metrics(y_test,  y_test_pred)

    print(f"\n{'─' * 50}")
    print(f"  {model_name}")
    print(f"{'─' * 50}")
    header = f"  {'Split':<8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    for split, m in [("Train", train_m), ("Test", test_m)]:
        print(f"  {split:<8} {m['accuracy']:>9.3f} {m['precision']:>10.3f} "
              f"{m['recall']:>8.3f} {m['f1']:>8.3f}")

    return {"train": train_m, "test": test_m}


# ═══════════════════════════════════════════════════════════════════════════
# 7. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate_models(X_train, X_test, y_train, y_test) -> tuple[dict, dict]:
    """
    Train four classifiers and evaluate each.

    The four models were chosen to represent a progression of complexity:
      - Perceptron:          simplest linear model, no regularisation
      - Naive Bayes:         probabilistic; strong TF-IDF baseline
      - LinearSVC:           margin-maximising; strong for sparse features
      - Logistic Regression: regularised probabilistic; calibrated outputs

    Returns:
        all_metrics:  {model_name: {"train": {...}, "test": {...}}}
        all_preds:    {model_name: test_predictions_array}
    """
    models = {
        "Perceptron":          Perceptron(),
        "Multinomial NB":      MultinomialNB(),
        "Linear SVC":          LinearSVC(),
        "Logistic Regression": LogisticRegression(max_iter=1500),
    }

    all_metrics = {}
    all_preds   = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)

        all_metrics[name] = report_results(name, y_train, train_pred, y_test, test_pred)
        all_preds[name]   = test_pred

    return all_metrics, all_preds


# ═══════════════════════════════════════════════════════════════════════════
# 8. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_length_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot histograms of review character lengths at three pipeline stages.

    Visualising length reduction confirms that each preprocessing step is
    having a measurable effect and that the pipeline isn't silently no-ops.
    Red dashed lines mark the mean at each stage.
    """
    stages = [
        (df["review_body"].apply(lambda x: len(str(x))),   "Raw Reviews",          "steelblue"),
        (df["cleaned_reviews"].apply(len),                  "After Cleaning",        "darkorange"),
        (df["processed_reviews"].apply(len),                "After Preprocessing",   "seagreen"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (lengths, title, color) in zip(axes, stages):
        ax.hist(lengths.clip(upper=1000), bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(lengths.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {lengths.mean():.0f}")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Character Length")
        ax.set_ylabel("Number of Reviews")
        ax.legend()

    plt.suptitle("Review Length Distribution Across Pipeline Stages",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}" if save_path else "")


def plot_wordclouds(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Generate word clouds for positive and negative reviews.

    Word clouds provide a fast qualitative sanity check: if the top words
    for positive reviews include sentiment noise (e.g., "the", "and"), the
    stop word removal step has a bug.
    """
    pos_text = " ".join(df[df["star_rating"] == 1]["processed_reviews"])
    neg_text = " ".join(df[df["star_rating"] == 0]["processed_reviews"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, text, title, cmap in zip(
        axes,
        [pos_text, neg_text],
        ["Positive Reviews", "Negative Reviews"],
        ["Greens", "Reds"],
    ):
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap=cmap, max_words=100).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold")

    plt.suptitle("Most Frequent Words by Sentiment Class",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}" if save_path else "")


def _draw_single_confusion_matrix(ax, y_true, y_pred, title: str) -> None:
    """Render a labelled confusion matrix on the provided Axes object."""
    cm = confusion_matrix(y_true, y_pred)
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Negative", "Positive"])

    threshold = cm.max() / 2
    for i, j in itertools.product(range(2), range(2)):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > threshold else "black")


def plot_confusion_matrices(all_preds: dict, y_test, save_path: str = None) -> None:
    """
    Plot confusion matrices for all models side by side.

    A confusion matrix reveals *how* a model fails:
      - High false negatives → model is too conservative (biased toward negative class)
      - High false positives → model is too permissive (biased toward positive class)
    Comparing across models shows whether different classifiers fail on the same examples.
    """
    n_models = len(all_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    for ax, (name, preds) in zip(axes, all_preds.items()):
        _draw_single_confusion_matrix(ax, y_test, preds, name)

    plt.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}" if save_path else "")


def plot_top_tfidf_features(vectorizer, X_train, y_train, top_n: int = 20,
                             save_path: str = None) -> None:
    """
    Bar chart of the top-N words by mean TF-IDF weight in each sentiment class.

    This analysis serves two purposes:
      1. Interpretability: shows which words most drive each sentiment
      2. Validation: confirms that preprocessing removed noise tokens;
         any stop words appearing here would flag a pipeline issue
    """
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Convert sparse matrix to dense for class-level averaging
    # Note: this is memory-intensive; only safe at this dataset scale
    X_dense     = X_train.toarray()
    y_train_arr = np.array(y_train)

    mean_pos = X_dense[y_train_arr == 1].mean(axis=0)
    mean_neg = X_dense[y_train_arr == 0].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, means, title, color in zip(
        axes,
        [mean_pos, mean_neg],
        [f"Top {top_n} Words — Positive", f"Top {top_n} Words — Negative"],
        ["seagreen", "crimson"],
    ):
        top_idx = means.argsort()[-top_n:][::-1]
        words   = feature_names[top_idx]
        values  = means[top_idx]
        # Reverse for bottom-to-top bar ordering
        ax.barh(words[::-1], values[::-1], color=color, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean TF-IDF Weight")

    plt.suptitle("Top TF-IDF Features by Sentiment Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}" if save_path else "")


def plot_model_accuracy_comparison(all_metrics: dict, save_path: str = None) -> None:
    """
    Bar chart comparing test accuracy across all four models.

    Provides a single-glance summary of relative model performance.
    """
    names   = list(all_metrics.keys())
    train_acc = [all_metrics[n]["train"]["accuracy"] for n in names]
    test_acc  = [all_metrics[n]["test"]["accuracy"]  for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, train_acc, width, label="Train", color="steelblue",   alpha=0.85)
    bars2 = ax.bar(x + width / 2, test_acc,  width, label="Test",  color="darkorange", alpha=0.85)

    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Test Accuracy by Model", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0.75, 1.02)
    ax.legend()
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)

    # Annotate bars with exact values
    for bar in list(bars1) + list(bars2):
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}" if save_path else "")


def plot_bigram_comparison(review_train, review_test, y_train, y_test,
                            save_path: str = None) -> None:
    """
    Compare Logistic Regression accuracy: unigrams vs unigrams+bigrams.

    Bigrams capture negation phrases ("not good", "does not work") that
    unigram models misclassify. The expected accuracy gain is modest
    (~0.5–1.5%) at the cost of a significantly larger feature space.
    """
    results = {}
    for label, ngram in [("Unigrams\n(1,1)", (1, 1)), ("Unigrams + Bigrams\n(1,2)", (1, 2))]:
        vec     = TfidfVectorizer(ngram_range=ngram)
        X_tr    = vec.fit_transform(review_train)
        X_te    = vec.transform(review_test)
        model   = LogisticRegression(max_iter=1500)
        model.fit(X_tr, y_train)
        results[label] = {
            "accuracy": accuracy_score(y_test, model.predict(X_te)),
            "features": X_tr.shape[1],
        }
        print(f"  {label.replace(chr(10), ' ')}: "
              f"accuracy={results[label]['accuracy']:.4f}, "
              f"features={results[label]['features']:,}")

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(results.keys())
    accs   = [results[k]["accuracy"] for k in labels]
    colors = ["steelblue", "seagreen"]

    bars = ax.bar(labels, accs, color=colors, edgecolor="white", alpha=0.85, width=0.4)
    ax.set_ylim(min(accs) - 0.01, max(accs) + 0.02)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("TF-IDF Unigrams vs Bigrams — Logistic Regression",
                 fontsize=12, fontweight="bold")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}" if save_path else "")


# ═══════════════════════════════════════════════════════════════════════════
# 9. SUGGESTED EXPERIMENTS (not executed — clearly marked)
# ═══════════════════════════════════════════════════════════════════════════

def suggested_experiment__cross_validation():
    """
    SUGGESTED EXPERIMENT — NOT EXECUTED.

    Replace the single 80/20 split with 5-fold stratified cross-validation to
    reduce variance in performance estimates and obtain confidence intervals.

    Pseudocode:
        from sklearn.model_selection import StratifiedKFold, cross_validate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for name, model in models.items():
            scores = cross_validate(model, X_all, y_all, cv=cv,
                                    scoring=["accuracy", "f1", "precision", "recall"])
            print(f"{name}: accuracy={scores['test_accuracy'].mean():.3f} "
                  f"± {scores['test_accuracy'].std():.3f}")
    """
    print("SUGGESTED EXPERIMENT: cross_validation — see docstring for implementation.")


def suggested_experiment__hyperparameter_tuning():
    """
    SUGGESTED EXPERIMENT — NOT EXECUTED.

    Grid search over regularisation strength C for Logistic Regression and
    LinearSVC; and smoothing alpha for Multinomial Naive Bayes.

    Expected outcome: 0.5–2% accuracy gain over defaults.

    Pseudocode:
        from sklearn.model_selection import GridSearchCV
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        gs = GridSearchCV(LogisticRegression(max_iter=1500), param_grid, cv=3,
                          scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"Best C: {gs.best_params_['C']}, CV accuracy: {gs.best_score_:.4f}")
    """
    print("SUGGESTED EXPERIMENT: hyperparameter_tuning — see docstring for implementation.")


def suggested_experiment__roc_auc():
    """
    SUGGESTED EXPERIMENT — NOT EXECUTED.

    Plot ROC curves for probability-outputting models (Logistic Regression,
    Multinomial NB). AUC measures separability independent of classification
    threshold — a model with AUC=0.95 but 90% accuracy may perform better in
    deployment if the operating threshold is shifted.

    Pseudocode:
        from sklearn.metrics import roc_curve, auc
        y_prob = lr_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"LR AUC = {auc(fpr, tpr):.3f}")
    """
    print("SUGGESTED EXPERIMENT: roc_auc — see docstring for implementation.")


def suggested_experiment__ablation_study():
    """
    SUGGESTED EXPERIMENT — NOT EXECUTED.

    Train Logistic Regression on progressively stripped versions of the
    pipeline to quantify each step's contribution to final accuracy.

    Steps to ablate (remove one at a time):
        - Without stop word removal
        - Without lemmatization
        - Without contraction expansion
        - Without HTML/URL removal
        - Raw text baseline (only lowercase)

    Expected outcome: a table showing accuracy drop for each ablation,
    justifying which preprocessing steps are most valuable.
    """
    print("SUGGESTED EXPERIMENT: ablation_study — see docstring for implementation.")


# ═══════════════════════════════════════════════════════════════════════════
# 10. MAIN ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Load ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  SENTIMENT ANALYSIS — Amazon Kitchen Reviews")
    print("=" * 60)

    fname = "amazon_reviews_us_Kitchen_v1_00.tsv.gz"
    df = load_data(fname)
    describe_raw_distribution(df)

    # ── Label and balance ───────────────────────────────────────────────────
    df = label_and_balance(df, n_per_class=100_000)

    # ── Clean ───────────────────────────────────────────────────────────────
    print("\n[Step 3] Cleaning text...")
    df = apply_cleaning(df)

    # ── Preprocess ──────────────────────────────────────────────────────────
    print("[Step 4] Preprocessing text...")
    df = apply_preprocessing(df)

    # ── Visualise pipeline effect ────────────────────────────────────────────
    print("\n[Viz 1] Review length distributions...")
    plot_length_distribution(df, save_path="review_length_distribution.png")

    print("[Viz 2] Word clouds...")
    plot_wordclouds(df, save_path="wordcloud.png")

    # ── Train / test split ───────────────────────────────────────────────────
    print("\n[Step 5] Splitting data (80/20)...")
    review_train, review_test, y_train, y_test = train_test_split(
        df["processed_reviews"], df["star_rating"],
        test_size=0.2, random_state=RANDOM_SEED, stratify=df["star_rating"]
    )

    # ── Feature extraction (unigram baseline) ────────────────────────────────
    print("[Step 6] Building TF-IDF features (unigrams)...")
    vectorizer, X_train, X_test = build_tfidf_features(review_train, review_test, ngram_range=(1, 1))

    # ── Train and evaluate all models ────────────────────────────────────────
    print("\n[Step 7] Training and evaluating models...")
    all_metrics, all_preds = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # ── Visualise model performance ──────────────────────────────────────────
    print("\n[Viz 3] Model accuracy comparison...")
    plot_model_accuracy_comparison(all_metrics, save_path="model_accuracy_comparison.png")

    print("[Viz 4] Confusion matrices...")
    plot_confusion_matrices(all_preds, y_test, save_path="confusion_matrices.png")

    print("[Viz 5] Top TF-IDF features per class...")
    plot_top_tfidf_features(vectorizer, X_train, y_train, top_n=20,
                             save_path="tfidf_top_features.png")

    # ── Bigram comparison ────────────────────────────────────────────────────
    print("\n[Viz 6] Unigram vs bigram comparison...")
    plot_bigram_comparison(review_train, review_test, y_train, y_test,
                            save_path="bigram_comparison.png")

    # ── Suggested experiments (not executed) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUGGESTED EXPERIMENTS (not executed)")
    print("=" * 60)
    suggested_experiment__cross_validation()
    suggested_experiment__hyperparameter_tuning()
    suggested_experiment__roc_auc()
    suggested_experiment__ablation_study()

    print("\nDone. All output files saved to working directory.")


if __name__ == "__main__":
    main()
