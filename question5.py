import pandas as pd
from sklearn.impute import SimpleImputer
from cleverminer import cleverminer
import io
import sys

# 1. Načtení předčištěného datasetu
df = pd.read_csv('newdataset_clean.csv', delimiter=';', engine='python', on_bad_lines='skip')
df.columns = df.columns.str.strip()

# 2. Odstranění záznamů bez Start_Time
df = df[~df['Start_Time'].isna()]
df = df[df['Start_Time'].astype(str).str.strip() != '']

# 3. Převedení Start_Time na datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='%d.%m.%Y %H:%M', errors='coerce')

# 4. Odstranění nevalidních časů
df = df.dropna(subset=['Start_Time'])

# 5. Rozřazení na den/noc
def get_time_of_day(start_time):
    if pd.isnull(start_time):
        return None
    hour = start_time.hour
    return 'Day' if 6 <= hour < 18 else 'Night'

df['time_of_day'] = df['Start_Time'].apply(get_time_of_day)

# 6. Odstranění záznamů s hodnotami mimo 'Day' nebo 'Night'
df = df[df['time_of_day'].isin(['Day', 'Night'])]

# 7. Vytvoření cílové proměnné severity_level
df['severity_level'] = df['Severity'].apply(lambda x: 'High' if int(x) >= 3 else 'Low')

# 8. Výběr pouze potřebných sloupců
df = df[['time_of_day', 'severity_level']]

# 9. Imputace (pro jistotu)
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=['time_of_day', 'severity_level'])

# Uložení vstupního datasetu pro kontrolu
df.to_csv("cfminer_question5_input.csv", index=False, sep=';')

# 10. Spuštění CF-Miner
clm = cleverminer(
    df=df,
    proc='CFMiner',
    target='severity_level',
    quantifiers={'conf': 0.95, 'Base': 10000},
    cond={
        'attributes': [
            {'name': 'time_of_day', 'type': 'subset', 'minlen': 1, 'maxlen': 1}

        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    }
)

# 11. Zachycení výstupu
output = io.StringIO()
sys.stdout = output

print("=== CF-Miner – Otázka 5: Jsou nehody vážnější v noci než ve dne? ===\n")
print("=== Shrnutí ===")
clm.print_summary()

print("\n=== Seznam pravidel ===")
clm.print_rulelist()
print("\n=== Všechna pravidla ===\n")
for i in range(1, len(clm.rulelist) + 1):
    print(f"\n=== Pravidlo {i} ===\n")
    clm.print_rule(i)


sys.stdout = sys.__stdout__

# 12. Uložení výstupu
with open("cfminer_question5_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("Výstup uložen jako cfminer_question5_output.txt")
