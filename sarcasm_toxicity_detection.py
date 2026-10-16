import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import squarify
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:\\Users\\gemin\\Downloads\\data sheet - research - Sheet1 (1).csv")  

print(df.columns)

# Drop the first and last column (index 0 and 6)
df = df.drop(columns=[df.columns[0], df.columns[-1]])

# Rename the remaining columns
df.columns = ['Platform', 'Comment', 'Toxicity', 'Sarcasm', 'Sentiment']

# View the cleaned column names
print("Final columns:", df.columns)

print(df.head())

# Strip whitespace, lowercase everything, then map to consistent values
df['Sentiment'] = df['Sentiment'].str.strip().str.lower()

# Map all values to consistent labels
df['Sentiment'] = df['Sentiment'].map({
    'negative': 'Negative',
    'neutral': 'Neutral',
    'positive': 'Positive'
})

# Optional: Check if any values didn't match and became NaN
print("Unique sentiments after cleaning:", df['Sentiment'].unique())

print(df['Sentiment'].value_counts())

#Count values in each column
print("Toxicity Counts:\n", df['Toxicity'].value_counts())
print("\nSarcasm Counts:\n", df['Sarcasm'].value_counts())
print("\nSentiment Counts:\n", df['Sentiment'].value_counts())

## Relationship Between Columns

# Toxicity vs Sarcasm
print("\nToxicity vs Sarcasm:\n", pd.crosstab(df['Toxicity'], df['Sarcasm']))

# Toxicity vs Sentiment
print("\nToxicity vs Sentiment:\n", pd.crosstab(df['Toxicity'], df['Sentiment']))

# Sarcasm vs Sentiment
print("\nSarcasm vs Sentiment:\n", pd.crosstab(df['Sarcasm'], df['Sentiment']))

#Print Percentages

print(f"Toxic Comments: {df['Toxicity'].mean() * 100:.2f}%")
print(f"Sarcastic Comments: {df['Sarcasm'].mean() * 100:.2f}%")

#Plot: Toxicity vs Sarcasm

sns.countplot(data=df, x='Platform', hue='Toxicity', palette='Set3')
plt.title("Toxicity by Platform")
plt.xlabel("Platform")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

##Heatmap of Correlations

# Create a numeric version of sentiment (optional)
sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
df['Sentiment_Num'] = df['Sentiment'].map(sentiment_map)

# Correlation heatmap
sns.heatmap(df[['Toxicity', 'Sarcasm', 'Sentiment_Num']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.show()

##Toxicity + Sarcasm Interaction by Platform

df['Toxic_and_Sarcastic'] = ((df['Toxicity'] == 1) & (df['Sarcasm'] == 1)).astype(int)

sns.countplot(data=df, x='Platform', hue='Toxic_and_Sarcastic', palette='flare')
plt.title("Toxic + Sarcastic Comments per Platform")
plt.xticks(rotation=45)
plt.legend(title="Toxic & Sarcastic")
plt.show()

toxic_comments = df[df['Toxicity'] == 1]['Comment'].dropna().astype(str).str.cat(sep=' ')
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(toxic_comments)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Toxic Comments")
plt.show()

#Treemap of Comment Distribution by Platform & Sentiment

treemap_data = df.groupby(['Platform', 'Sentiment']).size().reset_index(name='Counts')
labels = treemap_data.apply(lambda x: f"{x['Platform']}\n{x['Sentiment']}\n({x['Counts']})", axis=1)

plt.figure(figsize=(12, 7))
squarify.plot(sizes=treemap_data['Counts'], label=labels, alpha=0.8)
plt.axis('off')
plt.title("Comment Count by Platform and Sentiment")
plt.show()

treemap_data = df.groupby(['Platform', 'Sentiment']).size().reset_index(name='Counts')
labels = treemap_data.apply(lambda x: f"{x['Platform']}\n{x['Sentiment']}\n({x['Counts']})", axis=1)

plt.figure(figsize=(12, 7))
squarify.plot(sizes=treemap_data['Counts'], label=labels, alpha=0.8)
plt.axis('off')
plt.title("Comment Count by Platform and Sentiment")
plt.show()

# Convert sentiment to numeric for boxplot
df['Sentiment_Num'] = df['Sentiment'].map({'Negative': -1, 'Neutral': 0, 'Positive': 1})

sns.boxplot(data=df, x='Sarcasm', y='Sentiment_Num')
plt.title("Sentiment Spread in Sarcastic vs Non-Sarcastic Comments")
plt.xlabel("Sarcasm (0 = No, 1 = Yes)")
plt.ylabel("Sentiment Score")
plt.show()

sns.violinplot(data=df, x='Sentiment', y='Toxicity')
plt.title("Toxicity Distribution Across Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Toxicity (0–1)")
plt.show()

pivot = df.pivot_table(index='Platform', columns='Sentiment', values='Toxicity', aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap='coolwarm')
plt.title("Avg Toxicity by Platform and Sentiment")
plt.show()

# Drop rows where Toxicity or Sarcasm is missing
X = df[['Toxicity', 'Sarcasm']].dropna()

# Re-scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# If you want to add the clusters back to original dataframe:
df.loc[X.index, 'Cluster'] = clusters



sns.scatterplot(x='Toxicity', y='Sarcasm', hue='Cluster', data=df)
plt.title("Clusters of Toxicity & Sarcasm Behavior")
plt.show()

#How likely is a comment to be toxic given that it is sarcastic?


# P(Toxic | Sarcastic)
prob_toxic_given_sarcastic = df[(df['Sarcasm'] == 1) & (df['Toxicity'] == 1)].shape[0] / df[df['Sarcasm'] == 1].shape[0]
print(f"Probability of Toxicity given Sarcasm: {prob_toxic_given_sarcastic:.2f}")

#How do platform, sarcasm, and sentiment interact to affect toxicity?
pivot = df.pivot_table(index='Platform', columns='Sentiment', values='Toxicity', aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap='coolwarm')
plt.title("Avg Toxicity by Platform and Sentiment")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Convert sentiment to numeric
df['Sentiment_Num'] = df['Sentiment'].map({
    'Negative': -1,
    'Neutral': 0,
    'Positive': 1
})

# Select features and target, then drop rows with missing values
model_df = df[['Sarcasm', 'Sentiment_Num', 'Toxicity']].dropna()

X = model_df[['Sarcasm', 'Sentiment_Num']]
y = model_df['Toxicity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Print accuracy
print("Model accuracy:", model.score(X_test, y_test))


# Ensure Sentiment values are clean
df['Sentiment'] = df['Sentiment'].astype(str)

# Create subplots
sentiments = df['Sentiment'].unique()
fig, axes = plt.subplots(1, len(sentiments), figsize=(6 * len(sentiments), 5), sharey=True)

if len(sentiments) == 1:
    axes = [axes]  # Ensure axes is iterable if there's only one subplot

for ax, sentiment in zip(axes, sentiments):
    subset = df[df['Sentiment'] == sentiment]
    ct = pd.crosstab(subset['Toxicity'], subset['Sarcasm'])
    
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title(f"Sentiment: {sentiment}")
    ax.set_xlabel("Sarcasm")
    ax.set_ylabel("Toxicity")

plt.suptitle("Toxicity vs Sarcasm per Sentiment", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()
