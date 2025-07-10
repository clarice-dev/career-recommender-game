from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and encoders
with open('model/career_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/skills_encoder.pkl', 'rb') as f:
    skills_encoder = pickle.load(f)

with open('model/interests_encoder.pkl', 'rb') as f:
    interests_encoder = pickle.load(f)

# Home route
@app.route('/')
def home():
    education_levels = ['High School', "Bachelor's", "Master's", 'PhD']
    skills = list(skills_encoder.classes_)
    interests = list(interests_encoder.classes_)
    return render_template('quiz.html',
                           education_levels=education_levels,
                           skills=skills,
                           interests=interests)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    education = request.form.get('education')
    selected_skills = request.form.getlist('skills')
    selected_interests = request.form.getlist('interests')

    # One-hot encode education
    edu_df = pd.get_dummies([education], prefix='', prefix_sep='')
    all_edu_cols = ['High School', "Bachelor's", "Master's", 'PhD']
    for col in all_edu_cols:
        if col not in edu_df.columns:
            edu_df[col] = 0
    edu_df = edu_df[all_edu_cols]

    # Encode skills
    skills_vector = skills_encoder.transform([selected_skills])
    skills_df = pd.DataFrame(skills_vector, columns=skills_encoder.classes_)

    # Encode interests
    interests_vector = interests_encoder.transform([selected_interests])
    interests_df = pd.DataFrame(interests_vector, columns=interests_encoder.classes_)

    # Combine features
    X_user = pd.concat([edu_df, skills_df, interests_df], axis=1)

    # Ensure column order matches training
    for col in model.feature_names_in_:
        if col not in X_user.columns:
            X_user[col] = 0
    X_user = X_user[model.feature_names_in_]

    # Predict career
    career = model.predict(X_user)[0]
    return render_template('results.html', career=career)

if __name__ == '__main__':
    app.run(debug=True)