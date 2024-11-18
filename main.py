import pandas as pd
from preprocessing import preprocessing
from xg__boost import xgboost_model
from cat__boost import catboost
import tkinter as tk
from tkinter import font
from tkinter import messagebox
from randomforest import random_forest_model
import matplotlib.pyplot as plt
import numpy as np
# DataFrame di esempio
data = preprocessing()
df = pd.DataFrame({
    'nuova_tecnologia': pd.Categorical([]),
    'tipo_di_miglioramento_tecnologia': pd.Categorical([]),
    'continente': pd.Categorical([]),
    'stato': pd.Categorical([]),
    'budget_formazione': pd.Series(dtype='float'),
    'fatturato_conc(B)': pd.Series(dtype='float'),
    'finanziamenti': pd.Series(dtype='float'),
    'valore_startup': pd.Series(dtype='float'),
    'valore_mercato_totale': pd.Series(dtype='float'),
    'incremento_tec': pd.Series(dtype='float'),
    'media_fatturato': pd.Series(dtype='float'),
    'tasso_di_crescita_dip': pd.Series(dtype='float'),
    'numero_brevetti': pd.Series(dtype='int'),
    'numero_operatori': pd.Series(dtype='int'),
    'dip_ingegneri': pd.Series(dtype='int'),
    'dip_business': pd.Series(dtype='int'),
    'dip_vendite': pd.Series(dtype='int'),
    'dip_design': pd.Series(dtype='int'),
    'dip_informatica': pd.Series(dtype='int'),
    'dip_amministrativo': pd.Series(dtype='int'),
    'dip_controllo_qualità': pd.Series(dtype='int'),
    'dip_ricerca': pd.Series(dtype='int'),
    'dip_risorseumane': pd.Series(dtype='int'),
    'dip_assistenza': pd.Series(dtype='int')
})


# DataFrame di esempio per l'inserimento dei dati


# Otteniamo i valori unici per i campi da visualizzare nelle tendine
nuova_tecnologia = data['nuova_tecnologia'].unique()
settore_di_mercato = data['settore_di_mercato'].unique()
tipo_di_miglioramento_tecnologia = data['tipo_di_miglioramento_tecnologia'].unique()
continente = data['continente'].unique()

# Dizionario per memorizzare gli Entry e le selezioni delle tendine
entries = {}
selected_values = {}
multipliers = {}

# Funzione per creare un OptionMenu
def create_option_menu(campo, options, row):
    selected_values[campo] = tk.StringVar(root)
    selected_values[campo].set(options[0])  # Imposta il valore di default
    option_menu = tk.OptionMenu(root, selected_values[campo], *options)
    option_menu.grid(row=row, column=1, padx=10, pady=5)

# Funzione di callback per aggiornare gli stati in base al continente selezionato
def update_states(*args):
    continent_selected = selected_values['continente'].get()
    states = data[data['continente'] == continent_selected]['stato'].unique()
    selected_values['stato'].set(states[0])
    menu = option_menus['stato']['menu']
    menu.delete(0, 'end')
    for state in states:
        menu.add_command(label=state, command=lambda value=state: selected_values['stato'].set(value))

# Funzione per gestire la raccolta dei dati
def submit_data():
    global df
    error_messages = []
    dati = {}

    # Verifica le selezioni dei campi categorici
    for campo in ['nuova_tecnologia', 'tipo_di_miglioramento_tecnologia', 'continente', 'stato']:
        dati[campo] = selected_values[campo].get()

    # Verifica e raccogli i dati degli Entry
        for campo in entries.keys():
                try:
                    if campo in ['fatturato_conc(B)', 'incremento_tec', 'tasso_di_crescita_dip','valore_startup']:
                        dati[campo] = float(entries[campo].get())
                    else:
                        if campo in ['media_fatturato','finanziamenti','valore_mercato_totale']:
                            dati[campo] = float(entries[campo].get()) * int(multipliers[campo].get())
                        else:
                            dati[campo] = int(entries[campo].get())
                except ValueError:
                    error_messages.append(f"Il campo '{campo}' non è stato inserito correttamente.")
    if error_messages:
        messagebox.showerror("Errore", "\n".join(error_messages))
    else:
        try:
            # Aggiungere i dati al DataFrame
            new_row = pd.DataFrame([dati])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Visualizzare i dati in un messagebox
            messagebox.showinfo("Dati Inseriti", str(dati))
            
            # Chiudere la finestra principale
            root.destroy()
        except Exception as e:
            messagebox.showerror("Errore", f"Si è verificato un errore durante l'aggiunta dei dati.\n{e}")

# Funzione per confermare la chiusura della finestra
def on_closing():
    if messagebox.askokcancel("Conferma uscita", "Sei sicuro di voler chiudere?"):
        root.destroy()
# Creazione della finestra principale
root = tk.Tk()
root.title("Inserimento Dati")
root.state('zoomed')  # Imposta la finestra a schermo intero

# Aggiungi il protocollo per intercettare la chiusura della finestra
root.protocol("WM_DELETE_WINDOW", on_closing)

# Titolo
tk.Label(root, text="Inserimento dati Digital Twin", font=("Helvetica", 32)).grid(row=0, column=0, columnspan=5, pady=20)
tk.Label(root, text="Indicare con k le migliaia, con m i milioni e con b i miliardi", font=("Helvetica", 12)).grid(row=6, column=2, columnspan=5, pady=20)

# Configurazione delle colonne per espandersi con il contenuto
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_columnconfigure(3, weight=1)
root.grid_columnconfigure(4, weight=1)

# Campi da acquisire
campi = [
    'valore_startup', 'finanziamenti', 'budget_formazione', 
    'media_fatturato', 'valore_mercato_totale','fatturato_conc(B)', 'nuova_tecnologia',
    'tipo_di_miglioramento_tecnologia', 'incremento_tec', 'continente', 'stato', 'tasso_di_crescita_dip',
    'numero_brevetti','numero_operatori', 'dip_ingegneri', 'dip_business', 'dip_vendite', 'dip_design',
    'dip_informatica', 'dip_amministrativo', 'dip_controllo_qualità', 'dip_ricerca',
    'dip_risorseumane', 'dip_assistenza'
]

# Dizionario per memorizzare gli Entry e le selezioni delle tendine
entries = {}
selected_values = {}
option_menus = {}
multipliers = {}

# Creazione e disposizione dei widget
for i, campo in enumerate(campi):
    if campo in ['fatturato_conc(B)','tasso_di_crescita_dip','incremento_tec','numero_operatori', 'dip_ingegneri', 'dip_business', 'dip_vendite', 'dip_design','dip_informatica', 'dip_amministrativo', 'dip_controllo_qualità', 'dip_ricerca','dip_risorseumane', 'dip_assistenza','numero_brevetti']:
            
            if campo == 'fatturato_conc(B)':
                tk.Label(root, text="inserire fatturato concorrente in miliardi \n altrimenti se la cifra è inferiore al miliardo \n scrivere lo zero seguito dal punto e da cifre decimali").grid(row=i+1, column=0, padx=10, pady=5)
            else:
                if campo in  ['tasso_di_crescita_dip','incremento_tec']:
                    tk.Label(root,text="inserire "+campo+" in % \n senza inserire il simbolo").grid(row=i+1, column=0, padx=10, pady=5)
                else:    
                    tk.Label(root, text="inserire "+campo).grid(row=i+1, column=0, padx=10, pady=5)
            entries[campo] = tk.Entry(root)
            entries[campo].grid(row=i+1, column=1, padx=10, pady=5)
            
    else:
        tk.Label(root, text="inserire "+campo).grid(row=i+1, column=0, padx=10, pady=5)
    
        if campo in ['nuova_tecnologia', 'tipo_di_miglioramento_tecnologia', 'continente', 'stato']:
            options = data[campo].unique()
            create_option_menu(campo, options, i+1)
            if campo == 'continente':
                option_menus[campo] = tk.OptionMenu(root, selected_values[campo], *options)
                option_menus[campo].grid(row=i+1, column=1, padx=10, pady=5)
                selected_values[campo].trace("w", update_states)
            if campo == 'stato':
                option_menus[campo] = tk.OptionMenu(root, selected_values[campo], *options)
                option_menus[campo].grid(row=i+1, column=1, padx=10, pady=5)
        else:
            entries[campo] = tk.Entry(root)
            entries[campo].grid(row=i+1, column=1, padx=10, pady=5)
            
            # Variabile per il moltiplicatore
            multipliers[campo] = tk.StringVar(value="1")
            
            # Aggiunta dei Radiobutton B, M, K accanto all'Entry
            tk.Radiobutton(root, text="B", variable=multipliers[campo], value="1000000000").grid(row=i+1, column=2, padx=5)
            tk.Radiobutton(root, text="M", variable=multipliers[campo], value="1000000").grid(row=i+1, column=3, padx=5)
            tk.Radiobutton(root, text="K", variable=multipliers[campo], value="1000").grid(row=i+1, column=4, padx=5)


# Aggiunta di un pulsante per inviare i dati
my_font = font.Font(family="Helvetica", size=16)

# Creare il bottone con il font personalizzato
tk.Button(root, text="Invia", command=submit_data, font=my_font).grid(row=len(campi)-6, column=2, columnspan=5, pady=10)

# Avvia l'applicazione Tkinter
root.mainloop()
state_labels = {0: 'dead', 1: 'seed', 2:'series'}
nomi=['series','dead','seed']
america_data = data[(data['continente'] == 'North America')]
american_data = data[(data['continente'] == 'North America') & (data['stage']== 0)]
number1=len(american_data)
dead=data[(data['stage']==0)]
number2=len(dead)
number2=number2-number1
death=[number1,number2]
n4=len(america_data)
number3=len(data)-n4
numbers=[n4,number3]
stato_counts = america_data['stage'].value_counts()
stage_counts= data['stage'].value_counts()
sorted_indices = np.argsort(data['stage'])[::-1]
stato_labels = [state_labels[label] for label in stato_counts.index]
plt.figure(figsize=(8, 8))
plt.bar(range(len(stage_counts)), stage_counts, color="orange", align="center")
plt.xticks(range(len(stage_counts)),nomi , fontsize=8)
plt.title('distribuzioni per la classe target')
plt.savefig(f'images/preprocessing/distribuzione.png', bbox_inches='tight')
# Crea il grafico a torta
plt.figure(figsize=(8, 8))  # Imposta la dimensione della figura
plt.pie(death,labels=['Americane','di altri continenti'], autopct='%1.1f%%', startangle=140)
plt.title('confronto aziende fallite')
plt.savefig(f'images/preprocessing/aziende_attive_fallite.png', bbox_inches='tight')
plt.figure(figsize=(8, 8))  # Imposta la dimensione della figura
plt.pie(numbers,labels=['Americane','di altri continenti'], autopct='%1.1f%%', startangle=140)
plt.title('confronto aziende presenti nel dataset')
plt.savefig(f'images/preprocessing/dataset.png', bbox_inches='tight')
plt.figure(figsize=(8, 8))  # Imposta la dimensione della figura
plt.pie(stato_counts, labels=stato_labels, autopct='%1.1f%%', startangle=140, colors=['green', 'red','blue', 'orange', 'purple', 'brown', 'pink', 'yellow'])
plt.title('Distribuzione delle aziende Americane nel dataset')
plt.savefig(f'images/preprocessing/aziende_attive_fallite_america.png', bbox_inches='tight')
# Definire le feature categoriche e numeriche
categorical_features = ['nuova_tecnologia', 'tipo_di_miglioramento_tecnologia','stato','continente']
numeric_features = [
    'budget_formazione', 'fatturato_conc(B)', 'finanziamenti', 'valore_startup',
    'valore_mercato_totale', 'incremento_tec', 'media_fatturato', 'tasso_di_crescita_dip',
    'numero_operatori', 'dip_ingegneri', 'dip_business', 'dip_vendite', 'dip_design',
    'dip_informatica', 'dip_amministrativo', 'dip_controllo_qualità', 'dip_ricerca',
    'dip_risorseumane', 'dip_assistenza'
]
target = 'stage'
predxg=1
predcat=0
predrand=0
stage=['dead','seed','series']

if not df.empty:
# Eseguire il modello random forest    
    predrand= random_forest_model(data, categorical_features, numeric_features, target, df)
# Eseguire il modello XGBoost
    predxg = xgboost_model(data, categorical_features, numeric_features, target, df)
# Eseguire il modello CatBoost
    predcat = catboost(data, categorical_features, numeric_features, target, df)
    print(predcat)
    results_message = f"Predizione XGBoost: {stage[predxg]}\nPredizione CatBoost: {stage[predcat[0]]}\n Predizione Random forest: {stage[predrand]}\n"
    messagebox.showinfo("Risultati delle Predizioni", results_message)