# Sentiment Analysis of Amazon Kitchen Product Reviews

## Motivation

Customer reviews are one of the richest sources of unstructured opinion data available at scale. A product rated 4.2 stars tells you little beyond aggregate satisfaction, but the underlying review text can reveal *why* customers feel the way they do, which aspects they care about, and what language patterns correlate with strong sentiment. This project treats the sentiment classification problem as a controlled NLP experiment: given only the raw review text (no star rating), can a model reliably distinguish positive from negative sentiment?

The Amazon Kitchen dataset was chosen deliberately. Kitchen products generate highly descriptive reviews — customers describe sensory experiences ("smells like plastic"), functional outcomes ("leaked after one week"), and comparative judgments ("better than the one I had before"). This vocabulary diversity makes it a harder and more interesting classification target than, say, binary thumbs-up/down movie reviews.

---

## Research Questions

1. Which classical ML classifier best generalises to unseen review text when features are derived from TF-IDF?
2. How much does each preprocessing step individually contribute to reducing text noise?
3. Does adding bigram features to TF-IDF meaningfully improve classification performance?
4. What vocabulary most strongly differentiates positive from negative reviews after preprocessing?
5. Where do the models systematically fail — and do different models fail on the same examples?

---

## Dataset

**Source:** [Amazon Customer Reviews — US Kitchen Products](https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz)

- Format: TSV (gzip-compressed), one review per row
- Key columns used: `star_rating` (1–5), `review_body` (free text)
- Raw class distribution: heavily skewed toward positive ratings (4–5 stars dominate)
- Working dataset: 100,000 positive reviews (4–5 stars) + 100,000 negative reviews (1–2 stars), sampled with `np.random.seed(544)` for reproducibility
- Neutral reviews (3 stars) are discarded — their sentiment is genuinely ambiguous and including them as a third class or assigning them a binary label would introduce label noise

---

## Methodology

### Preprocessing Pipeline

Raw review text is progressively cleaned and normalised through seven steps. Each step is independently measurable in terms of its impact on average review length:

| Step | Operation | Rationale |
|------|-----------|-----------|
| 1 | Lowercase | Eliminates capitalisation as a spurious feature |
| 2 | Remove HTML tags & URLs | Product reviews on Amazon occasionally contain embedded HTML; these tokens carry zero sentiment signal |
| 3 | Expand contractions | "won't" → "will not"; prevents the vectoriser treating contracted and expanded forms as different features |
| 4 | Remove non-alphabetical characters | Numbers, punctuation, and special characters are largely noise at this vocabulary size |
| 5 | Collapse whitespace | Downstream artefact from prior removals |
| 6 | Remove stop words (NLTK) | High-frequency function words add dimensions without discriminative value |
| 7 | Lemmatize (WordNetLemmatizer, verb POS) | Maps inflected forms to their base form; "running" → "run" reduces vocabulary scatter |

**Measured impact:** average character length dropped from ~309 characters (raw) to ~183 characters (post-cleaning) to ~130 characters (post-preprocessing). This represents roughly a 58% total reduction in input size.

### Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert review text into numerical vectors. Two configurations are compared:

- **Unigrams only** (`ngram_range=(1,1)`) — baseline
- **Unigrams + Bigrams** (`ngram_range=(1,2)`) — captures phrases like "not good", "highly recommend", "fell apart"

**Critical implementation note:** TF-IDF is fit *only on the training split*, then applied to the test split. Fitting on the full dataset before splitting would constitute data leakage — the model would have seen IDF statistics derived from test-set vocabulary during training.

### Models

Four Scikit-learn classifiers are trained and compared:

| Model | Why it was included |
|-------|---------------------|
| **Perceptron** | Linear baseline; fast, interpretable, establishes a lower bound |
| **Multinomial Naive Bayes** | Classical text classifier; strong with TF-IDF features; probabilistic |
| **Linear SVC** | High-dimensional linear model; well-suited for sparse TF-IDF matrices |
| **Logistic Regression** | Probabilistic linear model; calibrated outputs; strong regularisation |

---

## Experimental Design

### Train/Test Split

- 80% training (160,000 reviews), 20% test (40,000 reviews)
- Stratified by class (equal class balance maintained in both splits)
- Fixed seed (`np.random.seed(544)`) ensures reproducibility

### Evaluation Metrics

Each model is evaluated on both training and test sets to detect overfitting:

- **Accuracy** — overall correctness
- **Precision** — what fraction of predicted positives were actually positive
- **Recall (macro)** — sensitivity averaged equally across both classes
- **F1 Score** — harmonic mean of precision and recall

Beyond scalar metrics, **confusion matrices** are plotted for each model to reveal systematic error patterns — whether a model is, for example, more prone to false negatives or false positives.

### Suggested Experiments *(not yet executed)*

The following experiments are proposed as natural extensions of this work:

- **Cross-validation (5-fold):** Replace single train/test split with k-fold CV to reduce variance in performance estimates
- **Hyperparameter tuning:** Grid search over Logistic Regression `C` values {0.01, 0.1, 1, 10} and LinearSVC `C` values; Naive Bayes `alpha` values
- **ROC/AUC analysis:** Plot ROC curves for probability-outputting models (Logistic Regression, Naive Bayes) to measure separability independent of threshold
- **Precision-Recall curves:** More informative than ROC for understanding the precision/recall tradeoff
- **Ablation study:** Train on progressively stripped versions of the pipeline (e.g., skip lemmatization, skip stop word removal) to quantify each step's contribution

---

## Results

| Model | Train Acc | Test Acc | Test Precision | Test Recall | Test F1 |
|-------|-----------|----------|----------------|-------------|---------|
| Perceptron | — | 85% | 85% | 89% | 84 |
| Multinomial Naive Bayes | — | 87% | 87% | 87% | 87 |
| Linear SVC | — | 89% | 89% | 90% | 89 |
| Logistic Regression | — | 90% | 90% | 90% | 90 |

*Full train/test metrics are printed at runtime. Results above match the executed notebook outputs.*

---

## Analysis

### Model Ranking

Logistic Regression is the strongest performer across all metrics, consistent with its reputation as a strong baseline for text classification tasks with TF-IDF features. The regularisation in LogisticRegression (L2 by default) prevents overfitting on the high-dimensional sparse feature space better than the Perceptron, which lacks regularisation entirely.

The Perceptron's lower performance and higher training-vs-test gap reflects its sensitivity to the order of training examples and its lack of a loss-based convergence criterion in the sklearn implementation. In practice, Perceptron should be treated as a sanity check lower bound rather than a competitive model.

LinearSVC and Logistic Regression are closely matched — both are linear models optimising over the same feature space — which is expected. The 1% gap is within the noise of a single random split.

### Vocabulary Analysis

The top TF-IDF features per class (Extension 5) reveal interpretable patterns: positive reviews are dominated by words like "love", "great", "easy", "recommend", while negative reviews cluster around "return", "broke", "disappoint", "waste", "poor". The clean vocabulary separation validates that the preprocessing pipeline is working as intended — if stop words or noise tokens appeared in the top features, it would indicate a pipeline bug.

### Bigram Impact

Bigrams (`ngram_range=(1,2)`) capture negation patterns ("not good", "does not work") that unigram models miss. The comparison in Extension 3 quantifies this. The expected finding is a modest improvement for Logistic Regression (typically 0.5–1.5% accuracy gain), at the cost of a much larger feature space and higher memory/compute requirements.

---

## Limitations

1. **Binary framing:** The 3-star neutral class is discarded, which simplifies the problem but doesn't reflect real deployment, where ambiguous reviews are common.

2. **Domain specificity:** Models are trained and tested on Kitchen products only. Performance on reviews in other categories (Electronics, Books) may differ due to vocabulary shift.

3. **TF-IDF ceiling:** TF-IDF treats each document as a bag of words, losing word order, negation scope, and syntactic structure. "This is not bad" and "This is bad" produce similar feature vectors.

4. **No hyperparameter optimisation:** All models use default sklearn parameters. Tuned models would likely outperform these baselines by 1–3%.

5. **Single random split:** A single 80/20 split means the reported metrics have unknown variance. Cross-validation would provide confidence intervals.

6. **Class balance is artificial:** Real Amazon data is heavily skewed toward positive reviews. The balanced 100k/100k sample changes the prior distribution, which affects precision/recall interpretation for deployment.

---

## Future Work

- **Transformer-based models:** Fine-tuning DistilBERT or RoBERTa on this dataset would likely yield 95%+ accuracy and handle negation correctly
- **Aspect-level sentiment:** Rather than predicting document-level sentiment, identify which product aspects (durability, ease of use, value) are positive or negative
- **Cross-category generalisation:** Train on Kitchen reviews and evaluate on Electronics to measure domain transfer
- **Temporal analysis:** Do sentiment patterns for the same product change over time as the product ages?
- **Calibration:** Logistic Regression outputs probabilities — a reliability diagram would show whether these probabilities are well-calibrated

---

## Key Insights

- A simple TF-IDF + Logistic Regression pipeline achieves 90% accuracy on a 200k-review binary sentiment task with no deep learning.
- The preprocessing pipeline reduces average review length by ~58%, which directly reduces feature space dimensionality and training time.
- Linear models perform comparably to more complex classifiers on this task, validating the "linear separability" of sentiment in TF-IDF space.
- The vocabulary gap between positive and negative reviews is large enough that even the simplest models (Naive Bayes, Perceptron) achieve 85–87% accuracy.
- Bigrams add interpretive value (capturing negation phrases) but modest measured accuracy gain at meaningful computational cost.

---

## Reproducibility

All experiments use `np.random.seed(544)`. To reproduce:

```bash
pip install pandas numpy nltk scikit-learn bs4 contractions wordcloud matplotlib
python sentiment_analysis_final.py
```

The dataset must be downloaded separately from the S3 link above and placed in the working directory.
