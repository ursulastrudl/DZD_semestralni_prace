import pandas as pd
from cleverminer import cleverminer
import io
import sys

# 1. Load data
df = pd.read_csv('newdataset_clean.csv', delimiter=';')


# 2. Clean necessary columns
df = df[df['Temperature(F)'].notna() & df['Severity'].notna() & df['Weather_Condition'].notna()]
df['Temperature(F)'] = pd.to_numeric(df['Temperature(F)'], errors='coerce')
df = df.dropna(subset=['Temperature(F)'])


# 3. Create target variable - udělat podrobnější kategorizaci, udělat levé a pravé řezy, délka 6, řez délka max 3
df['severity_level'] = df['Severity'].apply(lambda x: 'High' if int(x) >= 3 else 'Low')
def discretize_temp_6cat(temp):
    if temp < 20:
        return 1  # Extrémní mráz
    elif temp < 32:
        return 2  # Mírný mráz
    elif temp < 50:
        return 3  # Chladno
    elif temp < 70:
        return 4  # Mírné počasí
    elif temp < 85:
        return 5  # Teplo
    else:
        return 6  # Velké horko

df['Temperature_cat'] = df['Temperature(F)'].apply(discretize_temp_6cat)



def categorize_weather_numeric(condition):
    # 1) nejprve očistíme od whitespace
    cleaned = str(condition).strip()
    
    # 2) namapujeme na hlavní kategorie
    category_mapping = {
        # CLEAR
        'Fair': 'Clear',
        'Fair / Windy': 'Clear',

        # CLOUDY
        'Partly Cloudy': 'Cloudy',
        'Mostly Cloudy': 'Cloudy',
        'Cloudy': 'Cloudy',
        'Cloudy / Windy': 'Cloudy',
        'Mostly Cloudy / Windy': 'Cloudy',
        'Partly Cloudy / Windy': 'Cloudy',
            

        # RAIN
        'Rain': 'Rain',
        'Heavy Rain': 'Rain',
        'Light Rain': 'Rain',
        'Light Rain / Windy': 'Rain',
        'Light Drizzle': 'Rain',
        'Drizzle': 'Rain',
        'Drizzle / Windy': 'Rain',
        'Light Drizzle / Windy': 'Rain',
        'Heavy Rain / Windy': 'Rain',
        'Rain / Windy': 'Rain',
        'Light Rain Shower': 'Rain',
        'Heavy Drizzle': 'Rain',
        'Showers in the Vicinity': 'Rain',
        'Freezing Rain': 'Rain',
        'Freezing Rain / Windy': 'Rain',
        'Light Freezing Rain': 'Rain',
        'Freezing Drizzle': 'Rain',
        'Light Freezing Drizzle': 'Rain',

        # SNOW
        'Snow': 'Snow',
        'Light Snow': 'Snow',
        'Heavy Snow': 'Snow',
        'Wintry Mix': 'Snow',
        'Snow and Sleet': 'Snow',
        'Blowing Snow': 'Snow',
        'Light Snow / Windy': 'Snow',
        'Snow / Windy': 'Snow',
        'Heavy Snow / Windy': 'Snow',
        'Wintry Mix / Windy': 'Snow',
        'Light Snow and Sleet': 'Snow',
        'Snow and Sleet / Windy': 'Snow',
        'Blowing Snow / Windy': 'Snow',
        'Sleet': 'Snow',
        'Sleet / Windy': 'Snow',
        'Light Snow Shower': 'Snow',

        # STORM
        'T-Storm': 'Storm',
        'Thunder': 'Storm',
        'Heavy T-Storm': 'Storm',
        'Thunder in the Vicinity': 'Storm',
        'Thunder / Wintry Mix': 'Storm',
        'Thunder / Wintry Mix / Windy': 'Storm',
        'T-Storm / Windy': 'Storm',
        'Thunder / Windy': 'Storm',
        'Hail': 'Storm',
        'Light Rain with Thunder': 'Storm',
        'Light Snow with Thunder': 'Storm',

        # FOG
        'Fog': 'Fog',
        'Haze': 'Fog',
        'Mist': 'Fog',
        'Shallow Fog': 'Fog',
        'Patches of Fog': 'Fog',
        'Haze / Windy': 'Fog',
        'Fog / Windy': 'Fog',
        'Drizzle and Fog': 'Fog',
        'Widespread Dust': 'Fog',
        'Blowing Dust': 'Fog',
        'Blowing Dust / Windy': 'Fog',
        'N/A Precipitation': 'Fog'
    }

    category = category_mapping.get(cleaned, 'Unknown')

    # 3) čísla pro ordinalitu
    label_to_number = {
        'Clear': 1,
        'Cloudy': 2,
        'Rain': 3,
        'Snow': 4,
        'Fog': 5,
        'Storm': 6,
        'Unknown': None
    }

    return label_to_number[category]

 
df['Weather_numeric'] = df['Weather_Condition'].apply(categorize_weather_numeric)
# případně odfiltrovat Unknown:
df = df[df['Weather_numeric'].notna()]


# 4. Run SD4ft-Miner
clm = cleverminer(
    df=df,
    proc='SD4ftMiner',
    quantifiers={
        'Base1': 500,
        'Base2': 500,
        'Ratiopim': 1.1
    },
    ante={
        'attributes': [
            {'name': 'Weather_numeric', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
            {'name': 'State', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'minlen': 1,
        'maxlen': 2,
        'type': 'con'
    },
    succ={
        'attributes': [
            {'name': 'severity_level', 'type': 'one', 'value': 1, 'maxlen': 1}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    },
    frst={
        'attributes': [
            {'name': 'Temperature_cat', 'type': 'lcut', 'minlen': 1, 'maxlen': 3}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    },
    scnd={
        'attributes': [
            {'name': 'Temperature_cat', 'type': 'rcut', 'minlen': 1, 'maxlen': 3}
        ], 
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    }
)


# 5. Capture output
output = io.StringIO()
sys.stdout = output

print("=== SD4ft-Miner: Weather/State with Temp affecting Severity ===\n")
clm.print_summary()
print("\n=== Rules ===\n")
clm.print_rulelist()

# Výpis všech pravidel (v tomto případě 2)
for i in range(1, len(clm.rulelist) + 1):
    print(f"\n=== Rule {i} ===\n")
    clm.print_rule(i)

sys.stdout = sys.__stdout__

# 6. Save output
with open("sd4ftminer_question7_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("Výstup uložen jako sd4ftminer_question7_output.txt")
