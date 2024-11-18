import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, label_binarize
import shap
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import random
from sklearn.model_selection import StratifiedKFold

# Funzione per XGBoost
def xgboost_model(data, categorical, numeric, target, user_entries):
    # Separazione delle feature e del target
    X = data[categorical + numeric]
    y = data[target]
    stage=['dead','seed','series']
    # Codifica delle feature categoriche
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(data[categorical])
   
    # Creazione del DataFrame codificato
    encoded_columns = encoder.get_feature_names_out(categorical)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)
    
    # Concatenazione delle feature codificate con quelle numeriche
    X_final = pd.concat([X_encoded_df, data[numeric]], axis=1)
    
    # Suddivisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.35, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_test, label=y_test)
    evals = [(dtrain, 'train'), (dval, 'eval')]
    params = {
        'objective': 'multi:softmax',  # Per la classificazione multiclass
        'num_class': len(np.unique(y)), # Numero di classi
        'max_depth': 4,                 
        'eta': 0.1,                     
        'subsample': 0.8,               
        'colsample_bytree': 0.8,        
        'eval_metric': 'mlogloss'       
    }
    bst = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=10, evals=evals)
    optimal_estimators = bst.best_iteration
   

    # Creazione del modello
    model = XGBClassifier(
        n_estimators=optimal_estimators,
        learning_rate=0.1,
        max_depth=6,
        objective='multi:softmax',  # Cambia qui
        num_class=len(np.unique(y)), # Specifica il numero di classi
        eval_metric='mlogloss',     # Modifica la metrica di valutazione se necessario
        use_label_encoder=False
    )

    # Addestramento del modello
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100, early_stopping_rounds=50)

    # Predizioni sul test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Codifica dei dati di input dell'utente
    user_encoded = encoder.transform(user_entries[categorical])
    user_encoded_df = pd.DataFrame(user_encoded, columns=encoded_columns)
    
    # Concatenazione dei dati di input codificati con quelli numerici
    user_final = pd.concat([user_encoded_df, user_entries[numeric]], axis=1)
    
    # Predizione del target per i dati di input dell'utente
    y_user_pred = model.predict(user_final)
    y_user_prob = model.predict_proba(user_final)
    
 
    
    # Valutazione delle prestazioni
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  
    recall = recall_score(y_test, y_pred, average='macro')        
    f1 = f1_score(y_test, y_pred, average='macro')                
    logloss = log_loss(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Binarizzare le etichette per l'approccio one-vs-rest
    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    n_classes = y_test_binarized.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Valutazione del modello usando cross-validation
    scores = cross_val_score(model, X_final, y, cv=kf, scoring='accuracy')
    y_user_pred = y_user_pred.item()
    # Scrittura delle metriche su file
    try:
        with open('output.txt', 'a') as file:
            file.write("\nXGBoost\n")
            file.write(f"\naccuratezza: {scores.mean()}\n")
            file.write(f"Precisione: {precision}\n")
            file.write(f"Richiamo: {recall}\n")
            file.write(f"F1 Score: {f1}\n")
            file.write(f"Log Loss: {logloss}\n")
            file.write(f"Predizione per i dati dell'utente:"+stage[y_user_pred]+"\n")
            file.write(f"Probabilita' per i dati dell'utente:{y_user_prob}\n")
    except Exception as e:
        print(f"Errore durante la scrittura su file: {e}")

    # Riordinamento delle feature in base all'importanza
    feature_importances = model.feature_importances_
    feature_names = X_final.columns
    sorted_indices = np.argsort(feature_importances)[::-1]
    feature_importances_sorted = feature_importances[sorted_indices]
    feature_names_sorted = feature_names[sorted_indices]

    # Filtraggio delle feature con importanza non nulla
    feature_importances_nonzero = feature_importances_sorted[feature_importances_sorted > 0]
    feature_names_nonzero = feature_names_sorted[:len(feature_importances_nonzero)]

    # Creazione del grafico dell'importanza delle feature
    plt.figure()
    plt.title('Importanza delle Feature')
    plt.bar(range(len(feature_importances_nonzero)), feature_importances_nonzero, color="orange", align="center")
    plt.xticks(range(len(feature_names_nonzero)), feature_names_nonzero, rotation=90, fontsize=8)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig("images/xg_boost/xgb_feature.png")
    plt.close()

    # Creazione della matrice di confusione
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=stage,yticklabels=stage)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrice di Confusione')
    plt.savefig('images/xg_boost/confusion_matrix_xgb.png')
    plt.close()

    # Curva ROC
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve class {stage[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('images/xg_boost/roc_curvexgb.png')
    plt.close()
   
    # SHAP plot
    numero_casuale = random.randint(0, 9)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_final)
    for i in range(3):
        plt.figure()
        shap.plots.waterfall(shap_values[numero_casuale, :, i],show=False)
        plt.title(f"grafico per il target "+stage[i])
        plt.savefig(f'images/xg_boost/shap/waterfall_plot/'+stage[i]+'.png', bbox_inches='tight')
        plt.close()
        plt.figure()
        shap.summary_plot(shap_values[:,:,i], X_final,show=False)
        plt.title(f"grafico per il target "+stage[i])
        plt.savefig(f'images/xg_boost/shap/summary_plot/'+stage[i]+'.png', bbox_inches='tight')
        plt.close()
        plt.figure()
        shap.plots.bar(shap_values[:,:,i],show=False)
        plt.title(f"grafico per il target "+stage[i])
        plt.savefig(f'images/xg_boost/shap/barplot/'+stage[i]+'.png', bbox_inches='tight')
        plt.close()
    return y_user_pred
