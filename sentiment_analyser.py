import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('imdb_dataset.csv')
print("Sample Data:")
print(df.head())

x= df['review']
y= df['sentiment']
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42, stratify=y)

tfidf= TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
x_train_tfidf= tfidf.fit_transform(x_train)
x_test_tfidf= tfidf.transform(x_test)

model= LogisticRegression(max_iter=5000)
model.fit(x_train_tfidf, y_train)
y_pred= model.predict(x_test_tfidf)
accuracy_score= accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy_score)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
print(report_df)


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
