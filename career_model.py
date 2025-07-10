from sklearn.preprocessing import MultiLabelBinarizer #like one-hot encoding, but for lists
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle #let's you save python objects to files and load them later
import pandas as pd

# Load the dataset
df = pd.read_csv('AI-based Career.csv')

# Display basic information
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Sample data:")
print(df.head())

# Drop unused columns
df = df.drop(['CandidateID', 'Name', 'Recommendation_Score'], axis=1)
print("Columns:", df.columns.tolist())

# Split comma-separated strings into lists
df['Skills'] = df['Skills'].apply(lambda x: [skill.strip() for skill in x.split(',')])
df['Interests'] = df['Interests'].apply(lambda x: [i.strip() for i in x.split(',')])

# One-hot encode Skills and Interests
mlb_skills = MultiLabelBinarizer()
mlb_interests = MultiLabelBinarizer()

skills_encoded = pd.DataFrame(mlb_skills.fit_transform(df['Skills']), columns=mlb_skills.classes_)
interests_encoded = pd.DataFrame(mlb_interests.fit_transform(df['Interests']), columns=mlb_interests.classes_)

# One-hot encode Education
education_encoded = pd.get_dummies(df['Education'])

# Combine all features
X = pd.concat([education_encoded, skills_encoded, interests_encoded], axis=1)
y = df['Recommended_Career']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model and encoders
with open('career_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('skills_encoder.pkl', 'wb') as f:
    pickle.dump(mlb_skills, f)

with open('interests_encoder.pkl', 'wb') as f:
    pickle.dump(mlb_interests, f)

print("✅ Model trained and saved!")

# 🔍 Print unique career options
careers = df['Recommended_Career'].unique()
print(f"\n🎯 Total unique careers: {len(careers)}\n")
for i, career in enumerate(careers, 1):
    print(f"{i}. {career}")