import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test, vectorizer = joblib.load("vectorized_data.pkl")

print("🚀 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))

joblib.dump(model, "sentiment_model_rf.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\n💾 Model and vectorizer saved as sentiment_model_rf.pkl and tfidf_vectorizer.pkl")
