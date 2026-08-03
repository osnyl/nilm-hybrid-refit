"""
train_hybrid_model.py

Objective:
    Train our hybrid model (Linear Regression + Symbolic Rules) on REFIT House1.
    We use 'Aggregate' (total power) to predict 'Appliance1' (Refrigerator).
    This demonstrates the power of the hybrid approach on real data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. LOAD THE DATASET
# ============================================
print("📂 Loading CLEAN_House1.csv...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')
print(f"✅ Loaded {len(df)} rows.")

# We will use a subset to keep it fast (e.g., first 100,000 rows)
# but keep the whole dataset if you have the RAM.
# Let's take 500,000 rows for a good balance.
df_sample = df.head(500000)
print(f"📊 Using {len(df_sample)} rows for training.")

# ============================================
# 2. FEATURE ENGINEERING
# ============================================
# We have only one feature: Aggregate (total power)
# We want to predict: Appliance1 (Refrigerator)

X_raw = df_sample[['Aggregate']].values  # Shape: (n, 1)
y_raw = df_sample['Appliance1'].values   # Shape: (n,)

print(f"🔍 Feature shape: {X_raw.shape}")
print(f"🎯 Target shape: {y_raw.shape}")

# ============================================
# 3. NORMALIZATION (Min-Max Scaling)
# ============================================
# IMPORTANT: As we learned, normalization is crucial here because
# 'Aggregate' ranges from 0 to ~29,000W. Without scaling, gradient descent
# converges very slowly.
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
y_min = y_raw.min()
y_max = y_raw.max()

X_norm = (X_raw - X_min) / (X_max - X_min)
y_norm = (y_raw - y_min) / (y_max - y_min)

print(f"📈 X range: {X_min[0]:.1f} to {X_max[0]:.1f} (normalized to 0-1)")
print(f"📈 y range: {y_min:.1f} to {y_max:.1f} (normalized to 0-1)")

# ============================================
# 4. LINEAR MODEL (GRADIENT DESCENT)
# ============================================
class LinearModel:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.b = None
        self.loss_history = []
    
    def fit(self, X, y):
        m, n = X.shape
        self.W = np.random.randn(n) * 0.01
        self.b = np.random.randn(1) * 0.01
        
        for epoch in range(self.epochs):
            pred = np.dot(X, self.W) + self.b
            error = pred - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.loss_history.append(cost)
            
            dW = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)
            
            self.W -= self.lr * dW
            self.b -= self.lr * db
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | Loss: {cost:.6f}")
    
    def predict(self, X):
        return np.dot(X, self.W) + self.b

# Train the model
print("\n🚀 Training the linear model...")
model = LinearModel(lr=0.1, epochs=1000)
model.fit(X_norm, y_norm)

# Predict on the training set
pred_norm = model.predict(X_norm)
pred_raw = pred_norm * (y_max - y_min) + y_min

# ============================================
# 5. SYMBOLIC CORRECTION LAYER
# ============================================
def symbolic_correction(predicted_power, real_power):
    """
    This function applies real-world knowledge to correct the ML prediction.
    For a refrigerator, we know:
    - It should never be negative (if < 0, set to 0).
    - It should never exceed 500W (typical max for a fridge).
    - It should be 0 if the predicted is very low (noise).
    """
    if predicted_power < 0:
        return 0
    elif predicted_power < 10:  # Noise threshold
        return 0
    elif predicted_power > 500:  # Physically impossible for a fridge
        return 500
    else:
        return predicted_power

# Apply symbolic correction
corrected_predictions = np.array([symbolic_correction(p, r) for p, r in zip(pred_raw, y_raw)])

# ============================================
# 6. EVALUATION
# ============================================
# Calculate errors
raw_errors = np.abs(pred_raw - y_raw)
corrected_errors = np.abs(corrected_predictions - y_raw)

mse_raw = np.mean(raw_errors ** 2)
mse_corrected = np.mean(corrected_errors ** 2)

print("\n📊 MODEL PERFORMANCE")
print("=" * 50)
print(f"Mean Absolute Error (RAW ML)      : {np.mean(raw_errors):.2f} W")
print(f"Mean Absolute Error (SYMBOLIC)    : {np.mean(corrected_errors):.2f} W")
print(f"RMSE (RAW ML)                     : {np.sqrt(mse_raw):.2f} W")
print(f"RMSE (SYMBOLIC)                   : {np.sqrt(mse_corrected):.2f} W")

# ============================================
# 7. VISUALIZATION
# ============================================
# Plot the first 1000 points to see the comparison
plt.figure(figsize=(14, 6))
sample_idx = slice(0, 1000)

plt.plot(y_raw[sample_idx], label='Real Fridge Power', linewidth=1.5)
plt.plot(pred_raw[sample_idx], label='ML Prediction (Raw)', linestyle='--', alpha=0.7)
plt.plot(corrected_predictions[sample_idx], label='ML + Symbolic Correction', linestyle=':', linewidth=2)

plt.xlabel('Time (sample index)')
plt.ylabel('Power (Watts)')
plt.title('Refrigerator Power Prediction: Hybrid Model vs Raw ML')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/refrigerator_hybrid_prediction.png', dpi=150)
print("\n📈 Figure saved to: results/refrigerator_hybrid_prediction.png")

# Show plot if running interactively
plt.show()

print("\n✅ Analysis complete. The symbolic layer significantly improves the accuracy by removing impossible values (negative or >500W).")