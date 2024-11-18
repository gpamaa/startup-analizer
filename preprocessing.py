import pandas as pd

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
    
    #rimozione delle features inutili
    data = data.drop(columns=['id', 'nome', 'descrizione', 'città','concorrente'])
   
    data['finanziamenti']=data['finanziamenti'].astype(float)
    data['valore_startup']=data['valore_startup'].astype(float)
    data['valore_mercato_totale']=data['valore_mercato_totale'].astype(float)
    data['media_fatturato']=data['media_fatturato'].astype(float)
    data['tasso_di_crescita_dip']=data['tasso_di_crescita_dip'].astype(float)
    data['budget_formazione']=data['budget_formazione'].astype(float)
    data['nuova_tecnologia']=data['nuova_tecnologia'].astype('category')
    data['incremento_tec']=data['incremento_tec'].astype(float)
    data['fatturato_conc(B)']=data['fatturato_conc(B)'].astype(float)
    

# Conta quante aziende sono 'attive' e quante sono 'fallite'
    
    return data    