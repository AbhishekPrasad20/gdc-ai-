import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import random
import platform
import sys

# Print environment information for reproducibility
print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("scikit-learn not imported")

# Set fixed seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Parameters
MAX_FEATURES = 1000   # number of TF-IDF features
INPUT_FILE = "final_dataset.csv"
OUTPUT_FILE = "processed_dataset.csv"

# Load data
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} records from {INPUT_FILE}")

# Role → index mapping
role_to_idx = {"backend": 0, "frontend": 1, "fullstack": 2, "qa": 3}
df = df[df["role"].isin(role_to_idx.keys())]  # keep only valid roles
labels = df["role"].map(role_to_idx).values

# Numeric features
numeric_cols = ["numfileschanged", "linesadded", "linesdeleted", "numcommentsadded"]
numeric_feats = df[numeric_cols].astype(float).values

# Time features (hour of commit)
def extract_hour(x):
    try:
        return int(str(x).split(":")[0][-2:])
    except:
        return 0

hours = df["timeofcommit"].astype(str).apply(extract_hour).values.reshape(-1, 1)

# Commit message TF-IDF
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
tfidf_feats = vectorizer.fit_transform(df["commitmessage"].astype(str)).toarray()

# Combine all features
X = np.hstack([numeric_feats, hours, tfidf_feats])

# Save processed CSV
out = pd.DataFrame(X)
out["label"] = labels
out.to_csv(OUTPUT_FILE, index=False, header=False, float_format="%.6f")


print(f"Saved processed dataset: {OUTPUT_FILE}")
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
