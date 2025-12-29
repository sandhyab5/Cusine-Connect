from flask import Flask, render_template, request
import pickle
from utils.preprocess import preprocess_user_ingredients
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from ast import literal_eval
from googletrans import Translator

app = Flask(__name__)

# Get the directory of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one directory to access the data directory
data_dir = os.path.join(current_dir, "..", "data")

# Load all the pickle files using relative paths
recipes_path = os.path.join(data_dir, "processed", "recipes.pkl")
tfidf_matrix_path = os.path.join(data_dir, "processed", "tfidf_matrix.pkl")
vectorizer_path = os.path.join(data_dir, "processed", "vectorizer.pkl")

# Function to safely load pickle files
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

recipe_df = load_pickle(recipes_path)
tfidf_matrix = load_pickle(tfidf_matrix_path)
vectorizer = load_pickle(vectorizer_path)

translator = Translator()

# Safe translation function
def safe_translate(text, src_lang='en', dest_lang='te'):
    try:
        if text and text.strip():  # Ensure the text is not empty
            return translator.translate(text, src=src_lang, dest=dest_lang).text
        return ""
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return the original text if translation fails

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_ingredients = request.form.get('ingredients', '')
        if not user_ingredients.strip():
            return render_template('index.html', error="Please enter some ingredients.")

        # Preprocess user input and compute similarity
        preprocessed_user_ingredients = preprocess_user_ingredients(user_ingredients)
        user_ingredients_vector = vectorizer.transform([preprocessed_user_ingredients])
        similarity_scores = cosine_similarity(user_ingredients_vector, tfidf_matrix)
        top_indices = similarity_scores.argsort()[0][-5:][::-1]
        recommended_recipes = recipe_df.iloc[top_indices]

        # Parse and combine instructions
        try:
            recommended_recipes['instructions'] = recommended_recipes['instructions'].apply(literal_eval)
        except (ValueError, SyntaxError) as e:
            print(f"Error in parsing instructions: {e}")

        recommended_recipes['instructions_combined'] = recommended_recipes['instructions'].apply(
            lambda steps: " ".join(steps) if isinstance(steps, list) else ""
        )

        # Translate instructions to Telugu and Hindi
        recommended_recipes['instructions_te'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='te')
        )
        recommended_recipes['instructions_hi'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='hi')
        )
        recommended_recipes['instructions_kn'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='kn')
        )
        recommended_recipes['instructions_ta'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='ta')
        )
        recommended_recipes['instructions_ml'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='ml')
        )

        return render_template(
            'results.html',
            recipes=recommended_recipes,
            selected_language="en",
            selected_language_name="English",
            instructions_column="instructions_combined"
        )

    return render_template('index.html')

@app.route("/results", methods=['POST'])
def results():
    # Map the selected language to the respective instructions column
    language_map = {'en': "instructions_combined", 'te': "instructions_te", 'hi': "instructions_hi",'kn': "instructions_kn",'ta': "instructions_ta",'ml': "instructions_ml"}
    language_name_map = {'en': "English", 'te': "Telugu", 'hi': "Hindi",'kn': "Kannada",'ta': "Tamil",'ml': "Malayalam"}

    # Get the selected language from the form
    selected_language = request.form.get("language", "en")
    instructions_column = language_map.get(selected_language, "instructions_combined")
    selected_language_name = language_name_map.get(selected_language, "English")

    # Use the recipes already recommended (if stored in session or reprocess user input)
    user_ingredients = request.form.get('ingredients', '')
    if not user_ingredients.strip():
        return render_template('index.html', error="Please enter some ingredients.")

    # Preprocess user input and compute similarity
    preprocessed_user_ingredients = preprocess_user_ingredients(user_ingredients)
    user_ingredients_vector = vectorizer.transform([preprocessed_user_ingredients])
    similarity_scores = cosine_similarity(user_ingredients_vector, tfidf_matrix)
    # Print similarity scores for debugging
    print("\n=== Similarity Scores ===")
    print(similarity_scores)
    # Get top 5 recipe indices
    top_indices = similarity_scores.argsort()[0][-5:][::-1]

    # Print top 5 recommended recipes
    print("\n=== Top 5 Recommended Recipes ===")
    for idx in top_indices:
        print(f"Recipe Index: {idx}, Similarity Score: {similarity_scores[0][idx]}")

    recommended_recipes = recipe_df.iloc[top_indices]

    #top_indices = similarity_scores.argsort()[0][-5:][::-1]
    #recommended_recipes = recipe_df.iloc[top_indices]

    # Parse and combine instructions
    try:
        recommended_recipes['instructions'] = recommended_recipes['instructions'].apply(literal_eval)
    except (ValueError, SyntaxError) as e:
        print(f"Error in parsing instructions: {e}")

    recommended_recipes['instructions_combined'] = recommended_recipes['instructions'].apply(
        lambda steps: " ".join(steps) if isinstance(steps, list) else ""
    )

    # Translate instructions if not already translated
    if "instructions_te" not in recommended_recipes.columns:
        recommended_recipes['instructions_te'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='te')
        )
    if "instructions_hi" not in recommended_recipes.columns:
        recommended_recipes['instructions_hi'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='hi')
        )
    if "instructions_kn" not in recommended_recipes.columns:
        recommended_recipes['instructions_kn'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='kn')
        )
    if "instructions_ta" not in recommended_recipes.columns:
        recommended_recipes['instructions_ta'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='ta')
        )
    if "instructions_ml" not in recommended_recipes.columns:
        recommended_recipes['instructions_ml'] = recommended_recipes['instructions_combined'].apply(
            lambda text: safe_translate(text, src_lang='en', dest_lang='ml')
        )

    return render_template(
        'results.html',
        recipes=recommended_recipes,
        selected_language=selected_language,
        selected_language_name=selected_language_name,
        instructions_column=instructions_column
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port, debug=True)


