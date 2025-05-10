import pandas as pd
from cleverminer import cleverminer
from sklearn.impute import SimpleImputer
import io
import sys

# === 1. Načtení a agregace demografických dat ===
demography_raw = pd.read_csv('demographic_data.csv', delimiter=';')

# Agregace podle státu
state_population = demography_raw.groupby('State').agg({
    'Male Population': 'sum',
    'Female Population': 'sum',
    'Total Population': 'sum'
}).reset_index()

# Výpočet procentuálního zastoupení
state_population['Male (%)'] = (state_population['Male Population'] / state_population['Total Population'] * 100).round(2)
state_population['Female (%)'] = (state_population['Female Population'] / state_population['Total Population'] * 100).round(2)

# Uložení agregovaných dat
state_population.to_csv('demographic_data_grouped.csv', index=False, sep=';')

# === 2. Načtení dat o nehodách ===
accidents = pd.read_csv('newdataset_clean.csv', delimiter=';')

# Očištění názvů států (odstranění mezer navíc)
accidents['State Full'] = accidents['State Full '].str.strip()
state_population['State'] = state_population['State'].str.strip()

# === 3. Sloučení datasetů ===
merged_df = accidents.merge(state_population, left_on='State Full', right_on='State', how='left')

# Výběr důležitých sloupců (volitelné)
print(merged_df[['State Full', 'Male Population', 'Female Population', 'Total Population']].head())

# Uložení výsledku
merged_df.to_csv('newdataset_clean_with_pop_final.csv', index=False, sep=';')

#CLEVERMINER
# 1. Načtení dat
df = pd.read_csv('newdataset_clean_with_pop_final.csv', delimiter=';')
df['State'] = df['State Full'].str.strip()

# 2. Výběr relevantních sloupců
df = df[[
    'State',
    'Severity',
    'Male (%)',
    'Female (%)'
]]

# 3. Vytvoření cílové proměnné (High = Severity 3 nebo 4, jinak Low)
df['severity_level'] = df['Severity'].apply(lambda x: 'High' if int(x) >= 3 else 'Low')

# 4. Ošetření chybějících hodnot
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 5. Diskretizace podílů mužů/žen (kvartily)
df['Male (%)'] = pd.qcut(df['Male (%)'].astype(float), q=4, duplicates='drop').astype(str)
df['Female (%)'] = pd.qcut(df['Female (%)'].astype(float), q=4, duplicates='drop').astype(str)

# 6. Spuštění CleverMiner – 4ftMiner
clm = cleverminer(df=df,
    proc='4ftMiner',
    quantifiers={'Base': 10000, 'rel': 0.9},

    ante={
        'attributes': [
            {'name': 'Male (%)', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Female (%)', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'State', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'minlen': 1,
        'maxlen': 3,
        'type': 'con'
    },

    succ={
        'attributes': [
            {'name': 'severity_level', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'minlen': 1,
        'maxlen': 1,
        'type': 'con'
    }
)

# 7. Zachycení výstupu
output = io.StringIO()
sys.stdout = output

print("=== Shrnutí ===")
clm.print_summary()

print("\n=== Seznam pravidel ===")
clm.print_rulelist()


for i in range(1, len(clm.rulelist) + 1):
    print(f"\n=== Rule {i} ===\n")
    clm.print_rule(i)

sys.stdout = sys.__stdout__



# 8. Uložení do souboru
with open("4ftMiner_question2_output.txt", "w", encoding="utf-8") as f:
    f.write(output.getvalue())

print("Výstup uložen jako '4ftMiner_question2_output.txt'")