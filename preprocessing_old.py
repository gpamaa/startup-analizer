import pandas as pd
from sklearn.preprocessing import StandardScaler
def preprocessing():
    # Caricamento dei dati da un file CSV
    data = pd.read_csv('csv/main.csv')
    #Conversione dei dati in valori numerici accettabili
    data['budget_formazione'] = data['budget_formazione'].apply(lambda x: str(x).replace('K', '000'))
    data['budget_formazione'] = data['budget_formazione'].apply(lambda x: str(x).replace('M', '000000'))
    data['fatturato_conc(B)'] = data['fatturato_conc(B)'].apply(lambda x: str(x).replace('B', ''))
    data['finanziamenti'] = data['finanziamenti'].apply(lambda x: str(x).replace('M', '000000'))
    data['valore_startup'] = data['valore_startup'].apply(lambda x: str(x).replace('M', '000000'))
    data['valore_mercato_totale'] = data['valore_mercato_totale'].apply(lambda x: str(x).replace('B', '000000000'))
    data['valore_mercato_totale'] = data['valore_mercato_totale'].apply(lambda x: str(x).replace('M', '000000'))
    data['finanziamenti'] = data['finanziamenti'].apply(lambda x: str(x).replace('B', '000000000'))
    data['valore_startup'] = data['valore_startup'].apply(lambda x: str(x).replace('B', '000000000'))
    data['valore_startup'] = data['valore_startup'].apply(lambda x: str(x).replace('K', '000'))
    data['finanziamenti'] = data['finanziamenti'].apply(lambda x: float(str(x).replace('.', '')) * 0.1 if '.' in str(x) else float(x))
    data['valore_startup'] = data['valore_startup'].apply(lambda x: float(str(x).replace('.', '')) * 0.1 if '.' in str(x) else float(x))
    data['budget_formazione'] = data['budget_formazione'].apply(lambda x: float(str(x).replace('.', '')) * 0.1 if '.' in str(x) else float(x))
    data['valore_mercato_totale'] = data['valore_mercato_totale'].apply(lambda x: float(str(x).replace('.', '')) * 0.1 if '.' in str(x) else float(x))
    data['incremento_tec'] = data['incremento_tec'].apply(lambda x: str(x).replace('%', ''))
    data['media_fatturato'] = data['media_fatturato'].apply(lambda x: str(x).replace('M', '000000'))
    data['media_fatturato'] = data['media_fatturato'].apply(lambda x: str(x).replace('K', '000'))
    data['media_fatturato'] = data['media_fatturato'].apply(lambda x: str(x).replace('B', '000000000'))
    data['media_fatturato'] = data['media_fatturato'].apply(lambda x: float(str(x).replace('.', '')) * 0.1 if '.' in str(x) else float(x))
    data['tasso_di_crescita_dip'] = data['tasso_di_crescita_dip'].apply(lambda x: str(x).replace('%', ''))
    data['stage']=data['stage'].apply(lambda x: str(x).replace('dead', '0'))
    data['stage']=data['stage'].apply(lambda x: str(x).replace('seed', '1'))
    data['stage']=data['stage'].apply(lambda x: str(x).replace('series', '2'))
    data['stage']=data['stage'].astype(int)
    scaler = StandardScaler()
    #rimozione delle features inutili
    categorical_features = ['stage','nuova_tecnologia', 'tipo_di_miglioramento_tecnologia','stato','continente','settore_di_mercato']
    data_cat=data[categorical_features]
    data_cat2= pd.DataFrame(data_cat)
    data = data.drop(columns=[ 'stage','id','nome', 'descrizione', 'città','concorrente','nuova_tecnologia', 'tipo_di_miglioramento_tecnologia','stato','continente','settore_di_mercato'])
    nomi=data.columns
    dati_standardizzati = scaler.fit_transform(data)
    dati_standardizzati= pd.DataFrame(dati_standardizzati,columns=nomi)
    df_risultato = pd.concat([dati_standardizzati,data_cat2],axis=1)
    df_risultato['finanziamenti']=df_risultato['finanziamenti'].astype(float)
    df_risultato['valore_startup']=df_risultato['valore_startup'].astype(float)
    df_risultato['valore_mercato_totale']=df_risultato['valore_mercato_totale'].astype(float)
    df_risultato['media_fatturato']=df_risultato['media_fatturato'].astype(float)
    df_risultato['tasso_di_crescita_dip']=df_risultato['tasso_di_crescita_dip'].astype(float)
    df_risultato['budget_formazione']=df_risultato['budget_formazione'].astype(float)
    df_risultato['nuova_tecnologia']=df_risultato['nuova_tecnologia'].astype('category')
    df_risultato['incremento_tec']=df_risultato['incremento_tec'].astype(float)
    df_risultato['fatturato_conc(B)']=df_risultato['fatturato_conc(B)'].astype(float)

# Conta quante aziende sono 'attive' e quante sono 'fallite'
    
    return df_risultato