from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

importances = model.feature_importances_




from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)

import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x=importances, y=X_train.columns)
plt.title("Feature Importance (MDI)")
