#This ensures that the code is running without the UI/UX 

import pickle 
import pandas as pd

# Load model and encoders
with open('model/career_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/skills_encoder.pkl', 'rb') as f:
    skills_encoder = pickle.load(f)

with open('model/interests_encoder.pkl', 'rb') as f:
    interests_encoder = pickle.load(f)

# Function for numbered multi-select
def multi_select(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"{i + 1}. {opt}")
    selected = input("Enter numbers (comma separated): ").split(',')
    selected_clean = []
    for i in selected:
        i = i.strip()
        if i.isdigit() and 0 < int(i) <= len(options):
            selected_clean.append(options[int(i) - 1])
    return selected_clean

# Get Education Input
education_levels = ['High School', "Bachelor's", "Master's", 'PhD']
print("\n🎓 Select your education level:")
for i, level in enumerate(education_levels):
    print(f"{i + 1}. {level}")
edu_choice = int(input("Enter number: ")) - 1
education = education_levels[edu_choice]

# Get Skills Input
all_skills = list(skills_encoder.classes_)
user_skills = multi_select("💡 Select your skills:", all_skills)

# Get Interest Input
all_interests = list(interests_encoder.classes_)
user_interests = multi_select("❤️ Select your interests:", all_interests)

# Encode Inputs
# One hot encoding Education
edu_df = pd.get_dummies([education], prefix='', prefix_sep='')
all_edu_cols = education_levels
for col in all_edu_cols:
    if col not in edu_df.columns:
        edu_df[col] = 0
edu_df = edu_df[all_edu_cols]

# Encode skills
skills_vector = skills_encoder.transform([user_skills])
skills_df = pd.DataFrame(skills_vector, columns=skills_encoder.classes_)

# Encode interests
interests_vector = interests_encoder.transform([user_interests])
interests_df = pd.DataFrame(interests_vector, columns=interests_encoder.classes_)

# Combine features
X_user = pd.concat([edu_df, skills_df, interests_df], axis=1)

# Ensure all columns match training set
for col in model.feature_names_in_:
    if col not in X_user.columns:
        X_user[col] = 0
X_user = X_user[model.feature_names_in_]

# Predict career
prediction = model.predict(X_user)[0]
print(f"\n🎯 Recommended Career Path: {prediction}")