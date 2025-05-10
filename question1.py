# Existuje vztah mezi typickým počasím a typem nehody a závažností nehody?

import pandas as pd
from sklearn.impute import SimpleImputer
from cleverminer import cleverminer
import io
import sys

# 1. Načtení dat
df = pd.read_csv('newdataset_clean.csv', delimiter=';', engine='python', on_bad_lines='skip')

# 2. Výběr a přejmenování sloupců
df = df[['Weather_Condition', 'Temperature(F)', 'Humidity(%)',
         'Pressure(in)', 'Wind_Direction', 'Wind_Speed(mph)',
         'Precipitation(in)', 'Severity']]

df = df.rename(columns={
    'Temperature(F)': 'Temperature',
    'Humidity(%)': 'Humidity',
    'Pressure(in)': 'Pressure',
    'Wind_Speed(mph)': 'Wind_Speed',
    'Precipitation(in)': 'Precipitation'
})

# 3. Ošetření chybějících hodnot
imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 4. Převod Severity na kategorii High/Low
def classify_severity(sev):
    try:
        sev = int(sev)
        return 'High' if sev >= 3 else 'Low'
    except:
        return 'Low'

df['severity_level'] = df['Severity'].apply(classify_severity)

# 5. Kategorizace Weather_Condition na hlavní kategorie a číselné hodnoty
def categorize_weather_numeric(condition):
    cleaned = str(condition).strip()
    category_mapping = {
        'Fair': 'Clear', 'Fair / Windy': 'Clear',
        'Partly Cloudy': 'Cloudy', 'Mostly Cloudy': 'Cloudy', 'Cloudy': 'Cloudy',
        'Partly Cloudy / Windy': 'Cloudy', 'Mostly Cloudy / Windy': 'Cloudy', 'Cloudy / Windy': 'Cloudy',
        'Rain': 'Rain', 'Light Rain': 'Rain', 'Heavy Rain': 'Rain',
        'Light Rain / Windy': 'Rain', 'Light Drizzle': 'Rain', 'Drizzle': 'Rain',
        'Drizzle / Windy': 'Rain', 'Light Drizzle / Windy': 'Rain',
        'Heavy Rain / Windy': 'Rain', 'Rain / Windy': 'Rain',
        'Light Rain Shower': 'Rain', 'Heavy Drizzle': 'Rain', 'Showers in the Vicinity': 'Rain',
        'Freezing Rain': 'Rain', 'Freezing Rain / Windy': 'Rain',
        'Light Freezing Rain': 'Rain', 'Freezing Drizzle': 'Rain', 'Light Freezing Drizzle': 'Rain',
        'Snow': 'Snow', 'Light Snow': 'Snow', 'Heavy Snow': 'Snow', 'Wintry Mix': 'Snow',
        'Snow and Sleet': 'Snow', 'Blowing Snow': 'Snow', 'Light Snow / Windy': 'Snow',
        'Snow / Windy': 'Snow', 'Heavy Snow / Windy': 'Snow', 'Wintry Mix / Windy': 'Snow',
        'Light Snow and Sleet': 'Snow', 'Snow and Sleet / Windy': 'Snow',
        'Blowing Snow / Windy': 'Snow', 'Sleet': 'Snow', 'Sleet / Windy': 'Snow',
        'Light Snow Shower': 'Snow',
        'T-Storm': 'Storm', 'Thunder': 'Storm', 'Heavy T-Storm': 'Storm',
        'Thunder in the Vicinity': 'Storm', 'Thunder / Wintry Mix': 'Storm',
        'Thunder / Wintry Mix / Windy': 'Storm', 'T-Storm / Windy': 'Storm',
        'Thunder / Windy': 'Storm', 'Hail': 'Storm',
        'Light Rain with Thunder': 'Storm', 'Light Snow with Thunder': 'Storm',
        'Fog': 'Fog', 'Haze': 'Fog', 'Mist': 'Fog', 'Shallow Fog': 'Fog',
        'Patches of Fog': 'Fog', 'Haze / Windy': 'Fog', 'Fog / Windy': 'Fog',
        'Drizzle and Fog': 'Fog', 'Widespread Dust': 'Fog', 'Blowing Dust': 'Fog',
        'Blowing Dust / Windy': 'Fog', 'N/A Precipitation': 'Fog'
    }
    category = category_mapping.get(cleaned, 'Unknown')
    label_to_number = {'Clear': 1, 'Cloudy': 2, 'Rain': 3, 'Snow': 4, 'Fog': 5, 'Storm': 6, 'Unknown': None}
    return label_to_number[category]

df['Weather_numeric'] = df['Weather_Condition'].apply(categorize_weather_numeric)
df = df[df['Weather_numeric'].notna()]

# 6. Rozdělení numerických atributů do kvantilových kategorií
for col in ['Temperature', 'Humidity', 'Pressure', 'Wind_Speed', 'Precipitation']:
    df[col] = pd.qcut(df[col].astype(str).str.replace(',', '.').astype(float), q=4, duplicates='drop').astype(str)

# Visiblity, Wind_Chill nedávám - (Po konzultaci zjištěno, že nedává smysl)
# 7. Spuštění CleverMiner
clm = cleverminer(df=df,
    proc='4ftMiner',
    quantifiers={'Base': 700, 'rel': 0.9},

    ante={
        'attributes': [
            {'name': 'Weather_numeric', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Temperature', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Humidity', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Pressure', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Wind_Direction', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Wind_Speed', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Precipitation', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        ],
        'minlen': 1,
        'maxlen': 4,
        'type': 'con'
    },

    succ={
        'attributes': [
            {'name': 'severity_level', 'type': 'one', 'value': 'High'}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    }
)

# 8. Zachycení výstupu
output = io.StringIO()
sys.stdout = output

print("=== Shrnutí ===")
clm.print_summary()

print("\n=== Seznam pravidel ===")
clm.print_rulelist()

print("\n=== Jednotlivá pravidla ===")
for i in range(1, len(clm.rulelist) + 1):
    print(f"\n=== Rule {i} ===\n")
    clm.print_rule(i)

sys.stdout = sys.__stdout__


sys.stdout = sys.__stdout__

# 9. Uložení výstupu
with open("4ftMiner_question1_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("Výstup uložen jako 4ftMiner_question1_output.txt")