import pandas as pd
from cleverminer import cleverminer
import io
import sys

# 1. Načtení dat
df = pd.read_csv('newdataset_clean.csv', delimiter=';', engine='python', on_bad_lines='skip')
df.columns = df.columns.str.strip()

# 2. Čistění a výpočet rozdílu teploty a pocitové teploty
df = df[df['Temperature(F)'].notna() & df['Wind_Chill(F)'].notna() & df['Severity'].notna() & df['Weather_Condition'].notna() &df['city'].notna()]

df['Temperature(F)'] = pd.to_numeric(df['Temperature(F)'], errors='coerce')
df['Wind_Chill(F)'] = pd.to_numeric(df['Wind_Chill(F)'], errors='coerce')
df['Temp_Diff'] = df['Temperature(F)'] - df['Wind_Chill(F)']
df = df.dropna(subset=['Temp_Diff'])

print(df['Temp_Diff'].min(), df['Temp_Diff'].max())
print("Sloupce v datasetu:", df.columns.tolist())


# 3. Diskretizace rozdílu do 6 kategorií
bins = [0, 5, 10, 15, 20, 23, 26]
labels = [1, 2, 3, 4, 5, 6]

df['Temp_Diff_cat'] = pd.cut(df['Temp_Diff'], bins=bins, labels=labels, include_lowest=True)

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

# df['Humidity(%)'] = pd.to_numeric(df['Humidity(%)'], errors='coerce')
# df = df.dropna(subset=['Humidity(%)'])


# # Discretize humidity into 6 bins (quantiles)
# df['Humidity_cat'] = pd.qcut(df['Humidity(%)'], q=6, labels=[1, 2, 3, 4, 5, 6])

# # Wind Speed binning
# df['Wind_Speed(mph)'] = pd.to_numeric(df['Wind_Speed(mph)'], errors='coerce')
# df = df.dropna(subset=['Wind_Speed(mph)'])
# wind_bins = pd.qcut(df['Wind_Speed(mph)'], q=6, duplicates='drop')
# df['Wind_Speed_cat'] = pd.qcut(df['Wind_Speed(mph)'], q=6, labels=range(1, wind_bins.unique().size + 1), duplicates='drop')

# # Precipitation binning
# df['Precipitation(in)'] = pd.to_numeric(df['Precipitation(in)'], errors='coerce')
# df = df.dropna(subset=['Precipitation(in)'])
# precip_bins = pd.qcut(df['Precipitation(in)'], q=6, duplicates='drop')
# df['Precipitation_cat'] = pd.qcut(df['Precipitation(in)'], q=6, labels=range(1, precip_bins.unique().size + 1), duplicates='drop')

# df['hispanic'] = pd.to_numeric(df['demographic_data.Hispanic or Latino'], errors='coerce')
# df = df.dropna(subset=['hispanic'])
# precip_bins = pd.qcut(df['hispanic'], q=6, duplicates='drop')
# df['hispanic'] = pd.qcut(df['hispanic'], q=6, labels=range(1, precip_bins.unique().size + 1), duplicates='drop')


# Convert start_time to datetime and extract month
# Convert start_time to datetime and extract month
# df['start_time'] = pd.to_datetime(df['Start_Time'], errors='coerce', dayfirst=True)
# df = df.dropna(subset=['start_time'])  # Drop rows where parsing failed
# df['Month'] = df['start_time'].dt.month
# print("Numeric months:")
# print(df[['start_time', 'Month']].head())


# 4. Cílová proměnná
df['severity_level'] = df['Severity'].apply(lambda x: 'High' if int(x) >= 3 else 'Low')

# 5. Spuštění SD4ft-Miner
clm = cleverminer(
    df=df,
    proc='SD4ftMiner',
    quantifiers={
        'Base1': 100,
        'Base2': 100,
        'Ratiopim': 1.1
    },
    ante={
        'attributes': [
            {'name': 'Weather_numeric', 'type': 'seq', 'minlen': 1, 'maxlen': 2}
            
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    },
    succ={
        'attributes': [
            {'name': 'severity_level', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    },
    frst={
        'attributes': [
            {'name': 'Temp_Diff_cat', 'type': 'lcut', 'minlen': 1, 'maxlen': 3}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    },
    scnd={
        'attributes': [
            {'name': 'Temp_Diff_cat', 'type': 'rcut', 'minlen': 1, 'maxlen': 3}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    }

)

# 6. Výstup
output = io.StringIO()
sys.stdout = output

print("=== SD4ft-Miner – Vliv rozdílu teploty a pocitové teploty na závažnost nehody ===\n")
clm.print_summary()
print("\n=== Pravidla ===\n")
clm.print_rulelist()

for i in range(1, len(clm.rulelist) + 1):
    print(f"\n=== Rule {i} ===\n")
    clm.print_rule(i)

sys.stdout = sys.__stdout__

sys.stdout = sys.__stdout__

# 7. Uložení výstupu
with open("sd4ftminer_question_8_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("Výstup uložen jako sd4ftminer_question_8_output.txt")
