import pandas as pd

# Načti CSV
df = pd.read_csv('newdataset_clean.csv', delimiter=';', engine='python', on_bad_lines='skip')

# Seznam sloupců, které mají být čistě číselné
numeric_columns = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                   'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

# Odstranění řádků, kde některý ze sloupců obsahuje nečíselnou hodnotu
for col in numeric_columns:
    # Pokusí se převést hodnoty na čísla, pokud to nejde, nastaví NaN
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.strip(), errors='coerce')

# Odstraní řádky, kde je v některém z těchto sloupců NaN (tedy původně nečíselná hodnota)
df = df.dropna(subset=numeric_columns)


# Uložení vyčištěného datasetu
df.to_csv('newdataset_clean.csv', sep=';', index=False)
print("Vyčištěný dataset byl uložen jako newdataset_clean.csv")
