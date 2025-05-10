import pandas as pd
from sklearn.impute import SimpleImputer
from cleverminer import cleverminer
import io
import sys

# 1. Načtení CSV
df = pd.read_csv('newdataset_clean.csv', delimiter=';', engine='python', on_bad_lines='skip')
df.columns = df.columns.str.strip()  # ořez mezer

# 2. Definice infrastruktury - sloupečky, které se jí týkají
infra_cols = [
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
    'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
    'Traffic_Signal', 'Turning_Loop'
]

# 3. Výběr relevantních sloupců
df = df[infra_cols + ['Severity']]

# 4. Převod PRAVDA/NEPRAVDA na Yes/No
for col in infra_cols:
    df[col] = df[col].astype(str).str.upper().str.strip().map(lambda x: 'Yes' if x == 'PRAVDA' else 'No')

# 5. Vytvoření cílové proměnné "severity_level" - nad 3 je vysoká, pod 3 nízká
df['severity_level'] = df['Severity'].apply(lambda x: 'High' if int(x) >= 3 else 'Low')
df['severity_level'] = df['severity_level'].astype(str)

# 6. Vyčištění a imputace
df = df[infra_cols + ['severity_level']].dropna()
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=infra_cols + ['severity_level'])

# 7. Spuštění CF-Miner s target/value
clm = cleverminer(
    df=df,
    proc='CFMiner',
    target='severity_level',
    target_value='High',
    quantifiers={'conf': 0.95, 'Base': 10000},
    cond={
        'attributes': [
            {'name': col, 'type': 'subset', 'minlen': 1, 'maxlen': 1}
            for col in infra_cols
        ],
        'minlen': 1,
        'maxlen': 2,
        'type': 'con'
    }
)


# 8. Zachycení výstupu
output = io.StringIO()
sys.stdout = output

# Výpis všech pravidel podrobně

print("=== Shrnutí ===")
clm.print_summary()

print("\n=== Seznam pravidel ===")
clm.print_rulelist()

print("\n=== Detailní výpis všech pravidel ===\n")
for i in range(1, len(clm.rulelist) + 1):
    print(f"\n--- Pravidlo č. {i} ---\n")
    clm.print_rule(i)

# Ukončení přesměrování výstupu
sys.stdout = sys.__stdout__

# Uložení výstupu do souboru
with open("cfminer_question10_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("Výstup uložen jako cfminer_question10_output.txt")