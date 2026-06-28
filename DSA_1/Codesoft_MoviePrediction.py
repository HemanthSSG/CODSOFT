import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingRegressor
import os


def clean_year(year_val):
    if pd.isna(year_val):
        return np.nan
    val_str = str(year_val)
    digits = re.findall(r'\d+', val_str)
    if digits:
        return float(digits[0])
    return np.nan


def clean_duration(dur_val):
    if pd.isna(dur_val):
        return np.nan
    val_str = str(dur_val)
    digits = re.findall(r'\d+', val_str)
    if digits:
        return float(digits[0])
    return np.nan


def clean_votes(votes_val):
    if pd.isna(votes_val):
        return np.nan
    val_str = str(votes_val).replace(',', '')
    digits = re.findall(r'\d+', val_str)
    if digits:
        return float(digits[0])
    return np.nan


def main():
    print("=" * 60)
    print("        MOVIE RATING PREDICTOR MODEL TRAINING        ")
    print("=" * 60)

    # Path to the dataset
    csv_path = r"C:\Codesoft\DSA_1\Task 2(Movies list).csv"
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at '{csv_path}'.")
        print("Please place the dataset at the path above and try again.")
        return

    print(f"Loading dataset from: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Successfully loaded {df.shape[0]} rows.")

    # 1. Cleaning & Preprocessing
    print("Cleaning and preprocessing data...")
    # Drop rows without target Rating
    df = df.dropna(subset=['Rating']).reset_index(drop=True)

    # Parse numerical features
    df['Year_Cleaned'] = df['Year'].apply(clean_year)
    df['Duration_Cleaned'] = df['Duration'].apply(clean_duration)
    df['Votes_Cleaned'] = df['Votes'].apply(clean_votes)

    # Compute medians to fill missing values
    median_year = df['Year_Cleaned'].median()
    median_duration = df['Duration_Cleaned'].median()
    median_votes = df['Votes_Cleaned'].median()

    df['Year_Cleaned'] = df['Year_Cleaned'].fillna(median_year)
    df['Duration_Cleaned'] = df['Duration_Cleaned'].fillna(median_duration)
    df['Votes_Cleaned'] = df['Votes_Cleaned'].fillna(median_votes)

    # Create Votes_Log to handle skew
    df['Votes_Log'] = np.log1p(df['Votes_Cleaned'])

    # Process Genres
    genres_expanded = df['Genre'].str.split(', ', expand=True)
    unique_genres = set()
    for col in genres_expanded.columns:
        unique_genres.update(genres_expanded[col].dropna().unique())
    unique_genres = sorted(list(unique_genres))

    # One-hot encode individual genres
    for genre in unique_genres:
        df[f'Genre_{genre}'] = df['Genre'].fillna('').apply(lambda x: 1.0 if genre in x else 0.0)

    # Target Encoding and Frequency mapping for high-cardinality features
    high_card_cols = ['Director', 'Actor 1', 'Actor 2', 'Actor 3']
    target_mappings = {}
    freq_mappings = {}

    global_mean_rating = df['Rating'].mean()
    smoothing = 10

    for col in high_card_cols:
        # Save frequencies
        freq_map = df[col].value_counts().to_dict()
        freq_mappings[col] = freq_map

        # Calculate smoothed target encoding
        stats = df.groupby(col)['Rating'].agg(['mean', 'count'])
        smoothed_vals = (stats['mean'] * stats['count'] + global_mean_rating * smoothing) / (stats['count'] + smoothing)
        target_mappings[col] = smoothed_vals.to_dict()

        # Apply encoding to training df
        df[f'{col}_encoded'] = df[col].map(target_mappings[col]).fillna(global_mean_rating)
        df[f'{col}_freq'] = df[col].map(freq_mappings[col]).fillna(0)

    # Collective actor feature
    df['Actor_Mean_Rating'] = df[['Actor 1_encoded', 'Actor 2_encoded', 'Actor 3_encoded']].mean(axis=1)

    # Select features
    features = ['Year_Cleaned', 'Duration_Cleaned', 'Votes_Cleaned', 'Votes_Log', 'Actor_Mean_Rating']
    features += [f'Genre_{g}' for g in unique_genres]
    features += [f'{col}_encoded' for col in high_card_cols]
    features += [f'{col}_freq' for col in high_card_cols]

    X = df[features]
    y = df['Rating']

    # 2. Train Model
    print("Training Gradient Boosting Regressor (this may take a few seconds)...")
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("Model trained successfully!")

    print("\n" + "=" * 60)
    print("         INTERACTIVE MOVIE RATING PREDICTOR          ")
    print("=" * 60)
    print("Provide movie details below to predict its user/critic rating.")
    print("Press ENTER to use the default/median value where indicated.")
    print("-" * 60)

    while True:
        # Prompt user for inputs
        movie_name = input("\nEnter Movie Name [Optional]: ").strip()
        if not movie_name:
            movie_name = "Untitled Movie"

        # Year
        year_input = input(f"Enter Release Year [Default {int(median_year)}]: ").strip()
        if not year_input:
            year = median_year
        else:
            try:
                year = float(year_input)
            except ValueError:
                print(f"Invalid year! Using default: {int(median_year)}")
                year = median_year

        # Duration
        dur_input = input(f"Enter Duration in minutes [Default {int(median_duration)}]: ").strip()
        if not dur_input:
            duration = median_duration
        else:
            try:
                duration = float(dur_input)
            except ValueError:
                print(f"Invalid duration! Using default: {int(median_duration)}")
                duration = median_duration

        # Votes
        votes_input = input(f"Enter Number of Votes [Default {int(median_votes)}]: ").strip()
        if not votes_input:
            votes = median_votes
        else:
            try:
                votes = float(votes_input.replace(',', ''))
            except ValueError:
                print(f"Invalid votes! Using default: {int(median_votes)}")
                votes = median_votes

        votes_log = np.log1p(votes)

        # Genre
        genre_input = input("Enter Genre(s) (comma-separated, e.g. Drama, Romance) [Default: Drama]: ").strip()
        if not genre_input:
            genre_input = "Drama"

        # Director
        director = input("Enter Director Name [Default: Unknown]: ").strip()
        if not director:
            director = "Unknown"

        # Actors
        actor1 = input("Enter Lead Actor (Actor 1) [Default: Unknown]: ").strip()
        if not actor1:
            actor1 = "Unknown"

        actor2 = input("Enter Supporting Actor (Actor 2) [Default: Unknown]: ").strip()
        if not actor2:
            actor2 = "Unknown"

        actor3 = input("Enter Supporting Actor (Actor 3) [Default: Unknown]: ").strip()
        if not actor3:
            actor3 = "Unknown"

        # 3. Preprocess user input
        # Target Encoding
        dir_encoded = target_mappings['Director'].get(director, global_mean_rating)
        act1_encoded = target_mappings['Actor 1'].get(actor1, global_mean_rating)
        act2_encoded = target_mappings['Actor 2'].get(actor2, global_mean_rating)
        act3_encoded = target_mappings['Actor 3'].get(actor3, global_mean_rating)

        # Frequencies
        dir_freq = freq_mappings['Director'].get(director, 0)
        act1_freq = freq_mappings['Actor 1'].get(actor1, 0)
        act2_freq = freq_mappings['Actor 2'].get(actor2, 0)
        act3_freq = freq_mappings['Actor 3'].get(actor3, 0)

        # Actor mean rating
        actor_mean_rating = np.mean([act1_encoded, act2_encoded, act3_encoded])

        # Genre one-hot features
        genre_feats = {}
        for g in unique_genres:
            input_genres_list = [ig.strip().lower() for ig in genre_input.split(',')]
            if g.lower() in input_genres_list:
                genre_feats[f'Genre_{g}'] = 1.0
            else:
                genre_feats[f'Genre_{g}'] = 0.0

        # Build user feature dict
        user_data = {
            'Year_Cleaned': year,
            'Duration_Cleaned': duration,
            'Votes_Cleaned': votes,
            'Votes_Log': votes_log,
            'Actor_Mean_Rating': actor_mean_rating,
            'Director_encoded': dir_encoded,
            'Actor 1_encoded': act1_encoded,
            'Actor 2_encoded': act2_encoded,
            'Actor 3_encoded': act3_encoded,
            'Director_freq': dir_freq,
            'Actor 1_freq': act1_freq,
            'Actor 2_freq': act2_freq,
            'Actor 3_freq': act3_freq
        }
        for g in unique_genres:
            user_data[f'Genre_{g}'] = genre_feats[f'Genre_{g}']

        # Convert user_data to DataFrame matching column ordering
        user_df = pd.DataFrame([user_data])[features]

        # Predict Rating
        pred_rating = model.predict(user_df)[0]
        # Clip to valid rating range (1.0 to 10.0)
        pred_rating = np.clip(pred_rating, 1.0, 10.0)

        # Display Prediction Result
        print("\n" + "-" * 50)
        print(f" PREDICTION RESULT FOR: '{movie_name}'")
        print("-" * 50)
        print(f"  * Release Year      : {int(year)}")
        print(f"  * Duration          : {int(duration)} min")
        print(f"  * Votes             : {int(votes):,}")
        print(f"  * Genres            : {genre_input}")
        print(f"  * Director          : {director} (Historical Count: {dir_freq}, Encoded Rating: {dir_encoded:.2f})")
        print(f"  * Lead Actor        : {actor1} (Historical Count: {act1_freq}, Encoded Rating: {act1_encoded:.2f})")
        print(f"  * Actor 2           : {actor2} (Historical Count: {act2_freq}, Encoded Rating: {act2_encoded:.2f})")
        print(f"  * Actor 3           : {actor3} (Historical Count: {act3_freq}, Encoded Rating: {act3_encoded:.2f})")
        print("-" * 50)
        print(f" >>> ESTIMATED RATING  : {pred_rating:.2f} / 10.0 <<<")
        print("-" * 50)

        choice = input("\nDo you want to predict another movie's rating? (y/n) [Default: y]: ").strip().lower()
        if choice == 'n':
            print("\nThank you for using the Movie Rating Predictor. Goodbye!")
            break


if __name__ == "__main__":
    main()