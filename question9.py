import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Načíst data
df = pd.read_csv('newdataset_clean.csv', delimiter=';')
df.columns = df.columns.str.strip()

# 2. Odstranit chybějící hodnoty v rychlosti větru, tlaku a závažnosti
df = df[df['Wind_Speed(mph)'].notnull()]
df = df[df['Pressure(in)'].notnull()]
df = df[df['Severity'].notnull()]

# 3. Filtrovat pouze Severity == 2 nebo Severity == 4
df = df[df['Severity'].isin([2, 4])]

# 4. Převést Severity na dvě kategorie
def severity_label(sev):
    if sev == 2:
        return 'Nízká'
    elif sev == 4:
        return 'Vysoká'

df['Severity_Level'] = df['Severity'].apply(severity_label)

# 5. Převést sloupce na číselné (pokud jsou string s čárkou)
df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].astype(str).str.replace(',', '.').astype(float)
df['Pressure(in)'] = df['Pressure(in)'].astype(str).str.replace(',', '.').astype(float)

# 6. Příprava dat (vstupy: rychlost větru + tlak)
X = df[['Wind_Speed(mph)', 'Pressure(in)']]
y = df['Severity_Level']

# 7. Rozdělení na trénovací a testovací sadu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 8. Vytvoření a trénink K-NN modelu
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 9. Predikce
y_pred = knn.predict(X_test)

# 10. Vyhodnocení
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Vytisknout do konzole
print(cm)
print(cr)

# Uložit výstup do souboru
with open('K-NN-question9-output.txt', 'w', encoding='utf-8') as f:
    f.write('=== Confusion Matrix ===\n')
    f.write(str(cm))
    f.write('\n\n=== Classification Report ===\n')
    f.write(cr)

print("Výstup byl uložen do K-NN-question9-output.txt")

# 11. Vizualizace vstupních dat (scatter plot)
plt.scatter(X_train['Wind_Speed(mph)'], X_train['Pressure(in)'],
            c=y_train.map({'Nízká': 'blue', 'Vysoká': 'red'}))
plt.xlabel('Wind Speed (mph)')
plt.ylabel('Pressure (in)')
plt.title('Distribuce trénovacích dat podle závažnosti')
plt.savefig('K-NN-question9-scatter.png')
plt.close()
