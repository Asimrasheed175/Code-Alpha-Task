# ============================================================
# TASK 1: Iris Flower Classification
# CodeAlpha Data Science Internship
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Dataset ──────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 55)
print("       IRIS FLOWER CLASSIFICATION — CodeAlpha")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print(f"Classes       : {list(iris.target_names)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")
print(f"\nStatistical summary:\n{df.describe().round(2)}")

# ── 2. Exploratory Data Analysis ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Iris Flower — Exploratory Data Analysis", fontsize=16, fontweight='bold')

colors = {'setosa': '#E74C3C', 'versicolor': '#3498DB', 'virginica': '#2ECC71'}
species_colors = df['species'].map(colors)

# Scatter: sepal
ax = axes[0, 0]
for sp, col in colors.items():
    mask = df['species'] == sp
    ax.scatter(df.loc[mask, 'sepal length (cm)'],
               df.loc[mask, 'sepal width (cm)'],
               c=col, label=sp, alpha=0.7, edgecolors='white', s=60)
ax.set_xlabel('Sepal Length (cm)'); ax.set_ylabel('Sepal Width (cm)')
ax.set_title('Sepal: Length vs Width'); ax.legend()

# Scatter: petal
ax = axes[0, 1]
for sp, col in colors.items():
    mask = df['species'] == sp
    ax.scatter(df.loc[mask, 'petal length (cm)'],
               df.loc[mask, 'petal width (cm)'],
               c=col, label=sp, alpha=0.7, edgecolors='white', s=60)
ax.set_xlabel('Petal Length (cm)'); ax.set_ylabel('Petal Width (cm)')
ax.set_title('Petal: Length vs Width'); ax.legend()

# Violin plot per feature
ax = axes[0, 2]
df_melt = df.melt(id_vars='species',
                   value_vars=['sepal length (cm)', 'petal length (cm)'],
                   var_name='Feature', value_name='Value')
sns.violinplot(data=df_melt, x='Feature', y='Value', hue='species',
               ax=ax, palette=['#E74C3C','#3498DB','#2ECC71'])
ax.set_title('Feature Distributions by Species')
ax.set_xticklabels(['Sepal Len', 'Petal Len'])

# Histogram for each feature
features = iris.feature_names
for i, feat in enumerate(features[:3]):
    ax = axes[1, i]
    for sp, col in colors.items():
        ax.hist(df.loc[df['species'] == sp, feat], bins=15,
                alpha=0.6, color=col, label=sp)
    ax.set_xlabel(feat); ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution: {feat}'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/iris_eda.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 3. Correlation Heatmap ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/iris_corr.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 4. Train / Test Split & Scaling ─────────────────────────
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. Train Multiple Models ─────────────────────────────────
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42)
}

results = {}
print("\n" + "=" * 55)
print("  MODEL TRAINING & EVALUATION")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(model, X_train_sc, y_train, cv=5).mean()
    results[name] = {'accuracy': acc, 'cv_score': cv, 'model': model, 'y_pred': y_pred}
    print(f"\n{name}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  CV Score (5-fold): {cv:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]['accuracy'])
best      = results[best_name]
print(f"\n{'='*55}")
print(f"  BEST MODEL: {best_name}  —  Accuracy: {best['accuracy']*100:.2f}%")
print(f"{'='*55}")
print(f"\nDetailed Report ({best_name}):\n")
print(classification_report(y_test, best['y_pred'],
                             target_names=iris.target_names))

# ── 6. Confusion Matrix ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Results", fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, best['y_pred'])
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix\n{best_name}')

# Model comparison bar chart
model_names = list(results.keys())
accs = [results[m]['accuracy'] * 100 for m in model_names]
bar_colors = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6']
bars = axes[1].bar([m.replace(' ', '\n') for m in model_names], accs,
                   color=bar_colors, edgecolor='white', linewidth=1.2)
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[1].set_ylim(85, 102)
axes[1].set_ylabel('Test Accuracy (%)'); axes[1].set_title('Model Comparison')
axes[1].axhline(95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig('/home/claude/iris_results.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Task 1 Complete — plots saved.")
