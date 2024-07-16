import requests
import pandas as pd

# Function to fetch historical match data from football-data.org API
def fetch_historical_data(api_url, api_key):
    headers = {
        'X-Auth-Token': api_key
    }
    response = requests.get(api_url, headers=headers)
    data = response.json()
    return data

# Preprocess the historical match data
def preprocess_historical_data(data):
    matches = data['matches']
    df = pd.json_normalize(matches)
    print("DataFrame Columns:", df.columns)  # Print DataFrame columns to inspect
    df = df[['homeTeam.name', 'awayTeam.name', 'score.fullTime.homeTeam', 'score.fullTime.awayTeam', 'season.startDate']]
    df.columns = ['home_team', 'away_team', 'goals_home', 'goals_away', 'season_start_date']
    df.dropna(inplace=True)  # Drop rows with missing values
    return df

# Main function
def main():
    api_url = 'https://api.football-data.org/v2/competitions/PL/matches?season=2022'  # Premier League matches for 2022 season
    api_key = '434fa41dcc4c4c3db09193627521f910'  # Your actual API key
    data = fetch_historical_data(api_url, api_key)
    df = preprocess_historical_data(data)
    df.to_csv('historical_matches.csv', index=False)  # Save the historical data to a CSV file
    print(df.head())

if __name__ == "__main__":
    main()
