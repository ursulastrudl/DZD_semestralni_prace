#4. Má vyšší podíl černošské nebo hispánské populace vliv na závažnost nehod mimo okres Los Angeles

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Načíst data se správným oddělovačem
df = pd.read_csv('newdataset_clean.csv', delimiter=';')

# 2. Odstranit mezery v názvech sloupců
df.columns = df.columns.str.strip()

# 3. Filtrovat pouze nehody mimo Los Angeles
df = df[df['County'] != 'Los Angeles']

# 4. Předzpracování: vytvořit kategorie (binning)
df['Black_Bin'] = pd.cut(df['demographic_data.Black or African American Alone'], bins=3, labels=['Nízký', 'Střední', 'Vysoký'])
df['Hispanic_Bin'] = pd.cut(df['demographic_data.Hispanic or Latino'], bins=3, labels=['Nízký', 'Střední', 'Vysoký'])

# 5. Převést Severity na kategorie (pouze „Nízká“ a „Vysoká“)
def classify_severity(sev):
    try:
        sev = int(sev)
        return 'Vysoká' if sev >= 3 else 'Nízká'
    except:
        return 'Nízká'

df['Severity_Level'] = df['Severity'].apply(classify_severity)

# 6. Příprava X a y
X = df[['Black_Bin', 'Hispanic_Bin']]
y = df['Severity_Level']

# 7. Kódování vstupů na čísla
X_encoded = pd.get_dummies(X)

# 8. Trénink rozhodovacího stromu
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_encoded, y)

# 9. Uložit strom do souboru
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X_encoded.columns, class_names=['Nízká', 'Vysoká'], filled=True)
plt.savefig('decision_tree_question4.png')
plt.close()
