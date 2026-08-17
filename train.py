import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping

print("🚀 Training started")

# ================= PATH =================
BASE_DIR = r"C:\Users\purni\Desktop\pp1"
MODEL_DIR = os.path.join(BASE_DIR, "server", "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_PATH = r"C:\Users\purni\Desktop\cybercrime_forensic_dataset.xlsx"

# ================= LOAD DATA =================
df = pd.read_excel(DATASET_PATH)

# ================= TIMESTAMP =================
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df['Hour'] = df['Timestamp'].dt.hour
df['Day']  = df['Timestamp'].dt.day
df.fillna(0, inplace=True)

# ================= ENCODING =================
activity_encoder = LabelEncoder()
action_encoder   = LabelEncoder()
anomaly_encoder  = LabelEncoder()

df['Activity_Type'] = activity_encoder.fit_transform(df['Activity_Type'].astype(str))
df['Action']        = action_encoder.fit_transform(df['Action'].astype(str))
df['Anomaly_Type']  = anomaly_encoder.fit_transform(df['Anomaly_Type'].astype(str))

joblib.dump(activity_encoder, os.path.join(MODEL_DIR, "activity_encoder.pkl"))
joblib.dump(action_encoder,   os.path.join(MODEL_DIR, "action_encoder.pkl"))
joblib.dump(anomaly_encoder,  os.path.join(MODEL_DIR, "anomaly_encoder.pkl"))

# ================= LABEL =================
df['Label'] = df['Label'].map({'Normal': 0, 'Suspicious': 1})

# ================= FEATURES =================
features = [
    'Activity_Type',
    'Action',
    'Login_Attempts',
    'File_Size',
    'Anomaly_Type',
    'Hour',
    'Day'
]

X = df[features].values
y = df['Label'].values

# ================= SCALING =================
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# ================= LSTM SHAPE =================
# We reshape instead of fake repeating
X = X.reshape(X.shape[0], 1, X.shape[1])

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= CLASS WEIGHT (KEY FIX) =================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# ================= MODEL =================
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam

model = Sequential([
    LSTM(128, input_shape=(1, X.shape[2]), kernel_regularizer=L2(0.001), return_sequences=True),
    Dropout(0.2),
    LSTM(64, kernel_regularizer=L2(0.001), return_sequences=True),
    Dropout(0.2),
    LSTM(32, kernel_regularizer=L2(0.001)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=L2(0.001)),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=L2(0.001)),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model Summary:")
model.summary()

# ================= TRAIN =================
early_stop = EarlyStopping(patience=7, restore_best_weights=True, monitor='val_loss')

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ================= DISPLAY TRAINING PROGRESS IMMEDIATELY =================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2.5, color='#d62728')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5, color='#1f77b4')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Model Loss Over Epochs (During Training)', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2.5, color='#2ca02c')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2.5, color='#ff7f0e')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Model Accuracy Over Epochs (During Training)', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()  # Display while training completes
plt.savefig(os.path.join(graphs_dir, '01_training_history.png'), dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Training graph displayed and saved!")

# ================= SAVE =================
model.save(os.path.join(MODEL_DIR, "ddos_lstm_model.h5"))

print("✅ Training completed successfully")

# ================= PREDICTIONS =================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# ================= METRICS CALCULATION =================
# Standard Metrics
accuracy = accuracy_score(y_test, y_pred)

# Precision - Macro, Weighted, and Per-class
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)

# Recall - Macro, Weighted, and Per-class
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)

# F1-Score - Macro, Weighted, and Per-class
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

# For backward compatibility, keep binary versions
precision = precision_per_class[1] if len(precision_per_class) > 1 else precision_weighted
recall = recall_per_class[1] if len(recall_per_class) > 1 else recall_weighted
f1 = f1_per_class[1] if len(f1_per_class) > 1 else f1_weighted

roc_auc = roc_auc_score(y_test, y_pred_prob)
ap = average_precision_score(y_test, y_pred_prob)

# Calculate Specificity manually
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = recall  # Sensitivity is same as Recall

# Additional metrics
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

print("\n" + "="*70)
print("📊 COMPREHENSIVE MODEL METRICS - FINAL VALUES (WITH WEIGHTED ADJUSTMENT)")
print("="*70)
print("\n✓ BINARY CLASSIFICATION METRICS (Class 1: Suspicious)")
print("-"*70)
print(f"  Accuracy:                {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision (Class 1):     {precision:.4f}")
print(f"  Recall (Class 1):        {recall:.4f}")
print(f"  F1-Score (Class 1):      {f1:.4f}")
print(f"  Specificity:             {specificity:.4f}")
print(f"  True Positive Rate:      {true_positive_rate:.4f}")
print(f"  False Positive Rate:     {false_positive_rate:.4f}")
print(f"  ROC-AUC Score:           {roc_auc:.4f}")
print(f"  Average Precision:       {ap:.4f}")

print("\n✓ WEIGHTED METRICS (Accounts for Class Imbalance)")
print("-"*70)
print(f"  Precision (Weighted):    {precision_weighted:.4f}")
print(f"  Recall (Weighted):       {recall_weighted:.4f}")
print(f"  F1-Score (Weighted):     {f1_weighted:.4f}")

print("\n✓ MACRO METRICS (Equal weight for each class)")
print("-"*70)
print(f"  Precision (Macro):       {precision_macro:.4f}")
print(f"  Recall (Macro):          {recall_macro:.4f}")
print(f"  F1-Score (Macro):        {f1_macro:.4f}")

print("\n✓ PER-CLASS METRICS")
print("-"*70)
print(f"  Precision - Class 0 (Normal):      {precision_per_class[0]:.4f}")
print(f"  Precision - Class 1 (Suspicious):  {precision_per_class[1]:.4f}")
print(f"  Recall - Class 0 (Normal):         {recall_per_class[0]:.4f}")
print(f"  Recall - Class 1 (Suspicious):     {recall_per_class[1]:.4f}")
print(f"  F1-Score - Class 0 (Normal):       {f1_per_class[0]:.4f}")
print(f"  F1-Score - Class 1 (Suspicious):   {f1_per_class[1]:.4f}")

print("="*70)

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Suspicious']))

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n🔢 Confusion Matrix:")
print(cm)
print(f"\nConfusion Matrix Breakdown:")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")

# ================= GENERATE GRAPHS =================
graphs_dir = os.path.join(MODEL_DIR, "metrics_graphs")
os.makedirs(graphs_dir, exist_ok=True)

# 1. Training History Graph (Already displayed above, just reference)
print(f"✅ Saved: 01_training_history.png")

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
            xticklabels=['Normal', 'Suspicious'], yticklabels=['Normal', 'Suspicious'],
            annot_kws={'size': 14, 'fontweight': 'bold'})
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()  # Display
plt.savefig(os.path.join(graphs_dir, '02_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 02_confusion_matrix.png")

# 3. Metrics Comparison Bar Chart
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
plt.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()  # Display
plt.savefig(os.path.join(graphs_dir, '03_metrics_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 03_metrics_comparison.png")

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#1f77b4', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.show()  # Display
plt.savefig(os.path.join(graphs_dir, '04_roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 04_roc_curve.png")

# 4b. Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='#ff7f0e', lw=3, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '04b_precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 04b_precision_recall_curve.png")

# 5. Prediction Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_pred_prob[y_test == 0], bins=30, alpha=0.6, label='Normal (Actual)', color='green', edgecolor='black')
plt.hist(y_pred_prob[y_test == 1], bins=30, alpha=0.6, label='Suspicious (Actual)', color='red', edgecolor='black')
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
plt.xlabel('Prediction Probability', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '05_prediction_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 05_prediction_distribution.png")

# 6. Class Distribution
class_names = ['Normal', 'Suspicious']
class_counts = [np.sum(y_test == 0), np.sum(y_test == 1)]
colors_pie = ['#2ca02c', '#d62728']

plt.figure(figsize=(8, 6))
plt.pie(class_counts, labels=class_names, autopct='%1.1f%%', colors=colors_pie, startangle=90,
        explode=(0.05, 0.05), shadow=True, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Test Set Class Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '06_class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 06_class_distribution.png")

# 7. Comprehensive Metrics Table
fig, ax = plt.subplots(figsize=(14, 9))
ax.axis('tight')
ax.axis('off')

metrics_data = [
    ['Metric', 'Standard', 'Weighted', 'Macro', 'Description'],
    ['Accuracy', f'{accuracy:.4f}', '-', '-', 'Overall correctness of predictions'],
    ['Precision', f'{precision:.4f}', f'{precision_weighted:.4f}', f'{precision_macro:.4f}', 'True positives / All predicted positives'],
    ['Recall', f'{recall:.4f}', f'{recall_weighted:.4f}', f'{recall_macro:.4f}', 'True positives / All actual positives'],
    ['F1-Score', f'{f1:.4f}', f'{f1_weighted:.4f}', f'{f1_macro:.4f}', 'Harmonic mean of Precision & Recall'],
    ['Specificity', f'{specificity:.4f}', '-', '-', 'True negatives / All actual negatives'],
    ['ROC-AUC', f'{roc_auc:.4f}', '-', '-', 'Area under ROC curve'],
    ['Avg Precision', f'{ap:.4f}', '-', '-', 'Area under Precision-Recall curve'],
    ['True Positive Rate', f'{true_positive_rate:.4f}', '-', '-', 'TP / (TP + FN) - Sensitivity'],
    ['False Positive Rate', f'{false_positive_rate:.4f}', '-', '-', 'FP / (FP + TN)'],
]

table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.15, 0.15, 0.12, 0.33])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white', size=11)

# Alternate row colors
for i in range(1, len(metrics_data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Model Metrics Summary - Standard vs Weighted vs Macro', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '07_metrics_summary_table.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 07_metrics_summary_table.png")

# 8. Confusion Matrix Detailed Stats
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

cm_stats = [
    ['Metric', 'Value'],
    ['True Negatives (TN)', f'{tn}'],
    ['False Positives (FP)', f'{fp}'],
    ['False Negatives (FN)', f'{fn}'],
    ['True Positives (TP)', f'{tp}'],
    ['Total Samples', f'{len(y_test)}'],
    ['Sensitivity (TP / TP+FN)', f'{sensitivity:.4f}'],
    ['Specificity (TN / TN+FP)', f'{specificity:.4f}'],
    ['False Positive Rate', f'{false_positive_rate:.4f}'],
    ['False Negative Rate', f'{fn / (fn + tp) if (fn + tp) > 0 else 0:.4f}'],
]

cm_table = ax.table(cellText=cm_stats, cellLoc='center', loc='center',
                   colWidths=[0.5, 0.3])

cm_table.auto_set_font_size(False)
cm_table.set_fontsize(12)
cm_table.scale(1, 2.2)

# Style header
for i in range(2):
    cm_table[(0, i)].set_facecolor('#d62728')
    cm_table[(0, i)].set_text_props(weight='bold', color='white', size=13)

# Alternate colors
for i in range(1, len(cm_stats)):
    for j in range(2):
        if i % 2 == 0:
            cm_table[(i, j)].set_facecolor('#ffe6e6')
        else:
            cm_table[(i, j)].set_facecolor('white')

plt.title('Confusion Matrix - Detailed Statistics', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '08_confusion_matrix_stats.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 08_confusion_matrix_stats.png")

# 9. Extended Metrics Bar Chart (including new metrics)
extended_metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']
extended_metrics_values = [accuracy, precision, recall, specificity, f1, roc_auc]
extended_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#8c564b']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(extended_metrics_names, extended_metrics_values, color=extended_colors, 
              alpha=0.8, edgecolor='black', linewidth=2)

ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Extended Model Performance Metrics', fontsize=15, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, extended_metrics_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{value:.4f}\n({value*100:.2f}%)', ha='center', va='bottom', 
           fontweight='bold', fontsize=11)

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()  # Display
plt.savefig(os.path.join(graphs_dir, '09_extended_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 09_extended_metrics.png")

# 10. Weighted vs Macro vs Standard Metrics Comparison
fig, ax = plt.subplots(figsize=(13, 6))

metrics_comparison_names = ['Precision', 'Recall', 'F1-Score']
standard_vals = [precision, recall, f1]
weighted_vals = [precision_weighted, recall_weighted, f1_weighted]
macro_vals = [precision_macro, recall_macro, f1_macro]

x = np.arange(len(metrics_comparison_names))
width = 0.25

bars1 = ax.bar(x - width, standard_vals, width, label='Standard (Class 1)', 
               color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, weighted_vals, width, label='Weighted', 
               color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, macro_vals, width, label='Macro Average', 
               color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Precision, Recall & F1-Score Comparison: Standard vs Weighted vs Macro', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_comparison_names, fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '10_weighted_metrics_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 10_weighted_metrics_comparison.png")

# 11. Per-Class Metrics Comparison
fig, ax = plt.subplots(figsize=(12, 6))

class_names_metrics = ['Normal (Class 0)', 'Suspicious (Class 1)']
precision_classes = precision_per_class
recall_classes = recall_per_class
f1_classes = f1_per_class

x = np.arange(len(class_names_metrics))
width = 0.25

bars1 = ax.bar(x - width, precision_classes, width, label='Precision', 
               color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, recall_classes, width, label='Recall', 
               color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, f1_classes, width, label='F1-Score', 
               color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Per-Class Metrics: Precision, Recall & F1-Score', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names_metrics, fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '11_per_class_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: 11_per_class_metrics.png")

print(f"\n✅ All graphs saved to: {graphs_dir}")
print(f"📊 Total metric graphs generated: 11")
print(f"\n{'='*70}")
print("📁 Metrics Graph Files:")
print(f"{'='*70}")
print("  01_training_history.png - Loss and Accuracy over epochs")
print("  02_confusion_matrix.png - Confusion matrix heatmap")
print("  03_metrics_comparison.png - Basic metrics bar chart")
print("  04_roc_curve.png - ROC curve with AUC")
print("  04b_precision_recall_curve.png - Precision-Recall curve")
print("  05_prediction_distribution.png - Prediction probability distribution")
print("  06_class_distribution.png - Test set class distribution")
print("  07_metrics_summary_table.png - Comprehensive metrics table")
print("  08_confusion_matrix_stats.png - Detailed CM statistics")
print("  09_extended_metrics.png - Extended metrics comparison")
print("  10_weighted_metrics_comparison.png - Standard vs Weighted vs Macro")
print("  11_per_class_metrics.png - Per-class Precision/Recall/F1-Score")
print(f"{'='*70}")
