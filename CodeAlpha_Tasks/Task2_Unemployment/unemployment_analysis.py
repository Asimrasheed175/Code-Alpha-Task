# ============================================================
# TASK 2: Unemployment Analysis with Python
# CodeAlpha Data Science Internship
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("    UNEMPLOYMENT ANALYSIS — CodeAlpha")
print("=" * 60)

# ── 1. Synthetic Dataset (mirrors real patterns) ─────────────
# Monthly unemployment data for India (approx. CMIE-style)
# Covers Jan 2019 – Dec 2021 to capture pre/during/post-COVID
np.random.seed(42)
dates = pd.date_range(start='2019-01-01', end='2021-12-01', freq='MS')

# Pre-COVID baseline ~7-8%, COVID spike ~24%, recovery ~8-9%
unemp = []
for d in dates:
    if d < pd.Timestamp('2020-03-01'):          # pre-COVID
        u = np.random.normal(7.5, 0.6)
    elif d < pd.Timestamp('2020-06-01'):         # lockdown spike
        u = np.random.normal(21 + (d.month - 3) * 1.5, 1.2)
    elif d < pd.Timestamp('2020-10-01'):         # recovery
        u = np.random.normal(20 - (d.month - 5) * 1.8, 1.0)
    elif d < pd.Timestamp('2021-04-01'):         # new normal
        u = np.random.normal(9.5, 0.8)
    elif d < pd.Timestamp('2021-07-01'):         # 2nd wave
        u = np.random.normal(12, 1.0)
    else:                                         # stabilisation
        u = np.random.normal(8.5, 0.5)
    unemp.append(round(max(4, u), 2))

regions = ['Urban', 'Rural']
reg_data = []
for d, u in zip(dates, unemp):
    urban = round(u * np.random.uniform(1.05, 1.20), 2)
    rural = round(u * np.random.uniform(0.82, 0.97), 2)
    for region, val in zip(regions, [urban, rural]):
        reg_data.append({'Date': d, 'Region': region,
                         'Unemployment_Rate': val,
                         'Estimated_Unemployed': int(val * np.random.uniform(40, 60) * 1e5)})

df = pd.DataFrame(reg_data)
df_overall = df.groupby('Date')['Unemployment_Rate'].mean().reset_index()
df_overall.columns = ['Date', 'Overall_Rate']

print(f"\nDataset shape : {df.shape}")
print(f"Date range    : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Regions       : {df['Region'].unique().tolist()}")
print(f"\nSample data:\n{df.head(6).to_string(index=False)}")

# ── 2. Basic Stats ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  DESCRIPTIVE STATISTICS")
print("=" * 60)
for region in regions:
    sub = df[df['Region'] == region]['Unemployment_Rate']
    print(f"\n{region}:")
    print(f"  Mean={sub.mean():.2f}%  Median={sub.median():.2f}%"
          f"  Min={sub.min():.2f}%  Max={sub.max():.2f}%  Std={sub.std():.2f}%")

# COVID periods
pre  = df_overall[df_overall['Date'] < '2020-03-01']['Overall_Rate'].mean()
peak = df_overall[(df_overall['Date'] >= '2020-04-01') &
                  (df_overall['Date'] <= '2020-06-01')]['Overall_Rate'].mean()
post = df_overall[df_overall['Date'] >= '2021-06-01']['Overall_Rate'].mean()
print(f"\nCOVID-19 Impact:")
print(f"  Pre-COVID avg   : {pre:.2f}%")
print(f"  COVID peak avg  : {peak:.2f}%")
print(f"  Post-COVID avg  : {post:.2f}%")
print(f"  Spike increase  : +{peak - pre:.2f} percentage points")
print(f"  Recovery        : -{peak - post:.2f} pp from peak")

# ── 3. Visualisations ────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Unemployment Analysis — COVID-19 Impact Study\nCodeAlpha Data Science",
             fontsize=15, fontweight='bold', y=0.98)

# A) Time series — overall
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(df_overall['Date'], df_overall['Overall_Rate'],
         color='#2C3E50', linewidth=2.5, label='Overall Rate')
ax1.fill_between(df_overall['Date'], df_overall['Overall_Rate'],
                 alpha=0.15, color='#2C3E50')
# shade COVID waves
ax1.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-09-01'),
            alpha=0.18, color='red', label='Wave 1 Lockdown')
ax1.axvspan(pd.Timestamp('2021-03-01'), pd.Timestamp('2021-06-01'),
            alpha=0.18, color='orange', label='Wave 2')
ax1.axhline(pre,  linestyle='--', color='green', alpha=0.7, label=f'Pre-COVID avg ({pre:.1f}%)')
ax1.set_title('Overall Unemployment Rate Over Time', fontsize=12)
ax1.set_ylabel('Unemployment Rate (%)'); ax1.legend(fontsize=8)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

# B) Urban vs Rural
ax2 = fig.add_subplot(3, 2, 3)
for region, col in zip(regions, ['#E74C3C', '#27AE60']):
    sub = df[df['Region'] == region]
    ax2.plot(sub['Date'], sub['Unemployment_Rate'],
             label=region, color=col, linewidth=2, alpha=0.85)
ax2.set_title('Urban vs Rural Unemployment'); ax2.set_ylabel('Rate (%)')
ax2.legend(); ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

# C) Monthly average (seasonality)
df_overall['Month'] = df_overall['Date'].dt.month
monthly_avg = df_overall.groupby('Month')['Overall_Rate'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
ax3 = fig.add_subplot(3, 2, 4)
bars = ax3.bar(month_names, monthly_avg.values,
               color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 12)), edgecolor='white')
for bar, val in zip(bars, monthly_avg.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7)
ax3.set_title('Average Unemployment by Month (Seasonal Pattern)')
ax3.set_ylabel('Avg Rate (%)'); ax3.yaxis.set_major_formatter(mtick.PercentFormatter())

# D) Year-wise box plot
df_overall['Year'] = df_overall['Date'].dt.year
ax4 = fig.add_subplot(3, 2, 5)
df_overall.boxplot(column='Overall_Rate', by='Year', ax=ax4,
                   patch_artist=True,
                   boxprops=dict(facecolor='#3498DB', color='navy'),
                   medianprops=dict(color='red', linewidth=2))
ax4.set_title('Unemployment Distribution by Year')
ax4.set_ylabel('Rate (%)'); plt.sca(ax4); plt.title('Year-wise Distribution')

# E) COVID phase comparison bar
ax5 = fig.add_subplot(3, 2, 6)
phases   = ['Pre-COVID\n(2019–Feb 2020)', 'COVID Peak\n(Apr–Jun 2020)',
            'Post-COVID\n(H2 2021)']
values   = [pre, peak, post]
pal      = ['#27AE60', '#E74C3C', '#3498DB']
b = ax5.bar(phases, values, color=pal, edgecolor='white', linewidth=1.2, width=0.5)
for bar, v in zip(b, values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)
ax5.set_title('COVID-19 Phase Comparison'); ax5.set_ylabel('Avg Rate (%)')
ax5.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.tight_layout()
plt.savefig('/home/claude/unemployment_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 4. Trend analysis ────────────────────────────────────────
df_overall['Rolling_3m'] = df_overall['Overall_Rate'].rolling(3).mean()

print("\n" + "=" * 60)
print("  KEY INSIGHTS")
print("=" * 60)
print(f"1. COVID-19 caused unemployment to spike by {peak-pre:.1f} pp (peak vs pre-COVID)")
print(f"2. Urban areas showed consistently higher unemployment than rural.")
print(f"3. Unemployment peaked during Apr–May 2020 (national lockdown).")
print(f"4. Recovery began in Q3 2020 but a second wave in 2021 caused a mild spike.")
print(f"5. Post-COVID levels ({post:.1f}%) remain above pre-COVID ({pre:.1f}%).")
print(f"\n✅ Task 2 Complete — plot saved.")
