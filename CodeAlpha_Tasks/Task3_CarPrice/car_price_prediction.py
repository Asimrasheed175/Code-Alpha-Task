# ============================================================
# TASK 3: Car Price Prediction with Machine Learning
# CodeAlpha Data Science Internship
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("    CAR PRICE PREDICTION — CodeAlpha")
print("=" * 60)

# ── 1. Synthetic Dataset ─────────────────────────────────────
np.random.seed(42)
n = 1500

brands      = ['Toyota','Honda','BMW','Mercedes','Ford',
                'Hyundai','Volkswagen','Audi','Chevrolet','Kia']
brand_prem  = {'Toyota':1.0,'Honda':0.95,'BMW':2.4,'Mercedes':2.6,
               'Ford':0.9,'Hyundai':0.85,'Volkswagen':1.1,
               'Audi':2.2,'Chevrolet':0.88,'Kia':0.80}
fuel_types  = ['Petrol','Diesel','Electric','Hybrid']
trans_types = ['Manual','Automatic']
owners      = ['First','Second','Third']

brand_col   = np.random.choice(brands, n)
year        = np.random.randint(2008, 2023, n)
mileage     = np.random.uniform(5, 25, n)          # km/l
horsepower  = np.random.randint(60, 400, n)
engine_cc   = np.random.randint(800, 5000, n)
km_driven   = np.random.randint(1000, 200000, n)
fuel        = np.random.choice(fuel_types, n)
trans       = np.random.choice(trans_types, n)
owner       = np.random.choice(owners, n)

# Price formula with noise
age           = 2024 - year
brand_mult    = np.array([brand_prem[b] for b in brand_col])
fuel_mult     = np.where(fuel == 'Electric', 1.35,
                np.where(fuel == 'Diesel', 1.10,
                np.where(fuel == 'Hybrid', 1.20, 1.0)))
trans_mult    = np.where(trans == 'Automatic', 1.08, 1.0)
owner_mult    = np.where(owner == 'First', 1.0,
                np.where(owner == 'Second', 0.82, 0.68))

base_price = (horsepower * 220 + engine_cc * 1.5 + mileage * 8000
              - km_driven * 0.05 - age * 15000 + 150000)
price = (base_price * brand_mult * fuel_mult * trans_mult * owner_mult
         * np.random.uniform(0.88, 1.12, n))
price = np.clip(price, 50000, 12000000)

df = pd.DataFrame({
    'Brand': brand_col, 'Year': year, 'Mileage_kmpl': mileage.round(2),
    'Horsepower': horsepower, 'Engine_CC': engine_cc,
    'KM_Driven': km_driven, 'Fuel_Type': fuel,
    'Transmission': trans, 'Owner': owner,
    'Price': price.round(0).astype(int)
})

print(f"\nDataset shape : {df.shape}")
print(f"Price range   : ₹{df['Price'].min():,} — ₹{df['Price'].max():,}")
print(f"\nSample:\n{df.head(5).to_string(index=False)}")
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ── 2. EDA ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"\nCorrelation with Price:")
le = LabelEncoder()
df_enc = df.copy()
for col in ['Brand','Fuel_Type','Transmission','Owner']:
    df_enc[col] = le.fit_transform(df_enc[col])
corr = df_enc.corr()['Price'].drop('Price').sort_values(key=abs, ascending=False)
print(corr.round(3))

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Car Price Prediction — EDA\nCodeAlpha Data Science",
             fontsize=14, fontweight='bold')

# Scatter: HP vs Price
ax = axes[0, 0]
sc = ax.scatter(df['Horsepower'], df['Price']/1e5,
                c=df['Year'], cmap='viridis', alpha=0.4, s=20)
plt.colorbar(sc, ax=ax, label='Year')
ax.set_xlabel('Horsepower'); ax.set_ylabel('Price (₹ Lakh)')
ax.set_title('Horsepower vs Price (colored by Year)')

# Box: Brand
ax = axes[0, 1]
brand_order = df.groupby('Brand')['Price'].median().sort_values(ascending=False).index.tolist()
brand_data  = [df[df['Brand'] == b]['Price'].values / 1e5 for b in brand_order]
ax.boxplot(brand_data, patch_artist=True)
ax.set_xticks(range(1, len(brand_order)+1))
ax.set_xticklabels(brand_order, rotation=45, ha='right', fontsize=8)
ax.set_title('Price Distribution by Brand'); ax.set_ylabel('Price (₹ Lakh)')

# Bar: Fuel type avg price
ax = axes[0, 2]
fuel_avg = df.groupby('Fuel_Type')['Price'].mean() / 1e5
fuel_avg.sort_values(ascending=False).plot(kind='bar', ax=ax,
    color=['#E74C3C','#3498DB','#2ECC71','#F39C12'], edgecolor='white')
ax.set_title('Average Price by Fuel Type'); ax.set_ylabel('Avg Price (₹ Lakh)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for p in ax.patches:
    ax.text(p.get_x()+p.get_width()/2, p.get_height()+0.2,
            f'{p.get_height():.1f}L', ha='center', fontsize=8)

# Price distribution
ax = axes[1, 0]
ax.hist(df['Price']/1e5, bins=50, color='#3498DB', edgecolor='white', alpha=0.8)
ax.axvline(df['Price'].mean()/1e5, color='red', linestyle='--', label='Mean')
ax.axvline(df['Price'].median()/1e5, color='green', linestyle='--', label='Median')
ax.set_xlabel('Price (₹ Lakh)'); ax.set_ylabel('Count')
ax.set_title('Price Distribution'); ax.legend()

# KM driven vs Price
ax = axes[1, 1]
ax.scatter(df['KM_Driven']/1000, df['Price']/1e5, alpha=0.3, s=15, color='#9B59B6')
ax.set_xlabel('KM Driven (thousands)'); ax.set_ylabel('Price (₹ Lakh)')
ax.set_title('KM Driven vs Price')

# Correlation heatmap
ax = axes[1, 2]
num_cols = ['Year','Mileage_kmpl','Horsepower','Engine_CC','KM_Driven','Price']
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, center=0)
ax.set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('/home/claude/car_eda.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 3. Feature Engineering ───────────────────────────────────
df['Age']          = 2024 - df['Year']
df['Age_squared']  = df['Age'] ** 2
df['HP_per_CC']    = df['Horsepower'] / df['Engine_CC']

for col in ['Brand','Fuel_Type','Transmission','Owner']:
    df[col] = LabelEncoder().fit_transform(df[col])

features = ['Brand','Age','Age_squared','Mileage_kmpl','Horsepower',
            'Engine_CC','KM_Driven','Fuel_Type','Transmission','Owner','HP_per_CC']
X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Train Models ──────────────────────────────────────────
models = {
    'Linear Regression'      : LinearRegression(),
    'Ridge Regression'       : Ridge(alpha=10),
    'Decision Tree'          : DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest'          : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting'      : GradientBoostingRegressor(n_estimators=150, random_state=42)
}

results = {}
print("\n" + "=" * 60)
print("  MODEL TRAINING & EVALUATION")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2,
                     'model': model, 'y_pred': y_pred}
    print(f"\n{name}")
    print(f"  MAE  : ₹{mae:,.0f}")
    print(f"  RMSE : ₹{rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")

best_name = max(results, key=lambda k: results[k]['r2'])
best      = results[best_name]
print(f"\n{'='*60}")
print(f"  BEST MODEL: {best_name}  —  R² = {best['r2']:.4f}")
print(f"{'='*60}")

# ── 5. Result Plots ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(f"Car Price Prediction Results\nBest Model: {best_name}",
             fontsize=13, fontweight='bold')

# Actual vs Predicted
ax = axes[0]
ax.scatter(y_test/1e5, best['y_pred']/1e5, alpha=0.3, s=15, color='#3498DB')
lim = [min(y_test.min(), best['y_pred'].min())/1e5,
       max(y_test.max(), best['y_pred'].max())/1e5]
ax.plot(lim, lim, 'r--', linewidth=2, label='Perfect fit')
ax.set_xlabel('Actual Price (₹ Lakh)'); ax.set_ylabel('Predicted Price (₹ Lakh)')
ax.set_title('Actual vs Predicted'); ax.legend()

# R² comparison
ax = axes[1]
r2s   = [results[m]['r2'] for m in results]
names = [m.replace(' ', '\n') for m in results]
cols  = ['#E74C3C' if m == best_name else '#3498DB' for m in results]
b = ax.bar(names, r2s, color=cols, edgecolor='white')
for bar, v in zip(b, r2s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{v:.3f}', ha='center', fontsize=8)
ax.set_ylim(0, 1.05); ax.set_ylabel('R² Score'); ax.set_title('Model R² Comparison')

# Feature Importance (RF)
ax = axes[2]
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=features)
importances.sort_values().tail(8).plot(kind='barh', ax=ax, color='#27AE60', edgecolor='white')
ax.set_title('Feature Importances\n(Random Forest)'); ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig('/home/claude/car_results.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Task 3 Complete — plots saved.")
