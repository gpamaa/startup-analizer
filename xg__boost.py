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
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Funzione per XGBoost
def xgboost_model(data, categorical, numeric, target, user_entries):
    # Separazione delle feature e del target
    X = data[categorical + numeric]
    y = data[target]
    stage=['dead','seed','series']
    empty= [134,142,143,154]
    # Codifica delle feature categoriche
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(data[categorical])
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    tprs = {i: [] for i in range(n_classes)}
    fprs = {i: [] for i in range(n_classes)}
    aucs = {i: [] for i in range(n_classes)}
    # Creazione del DataFrame codificato
    encoded_columns = encoder.get_feature_names_out(categorical)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)
    
    # Concatenazione delle feature codificate con quelle numeriche
    X_final = pd.concat([X_encoded_df, data[numeric]], axis=1)
    #ros = RandomOverSampler()
    #X_resampled, y_resampled = ros.fit_resample(X_final, y)
    smote = SMOTE()
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3)
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

    
    # Codifica dei dati di input dell'utente
    user_encoded = encoder.transform(user_entries[categorical])
    user_encoded_df = pd.DataFrame(user_encoded, columns=encoded_columns)
    
    # Concatenazione dei dati di input codificati con quelli numerici
    user_final = pd.concat([user_encoded_df, user_entries[numeric]], axis=1)
    
    # Predizione del target per i dati di input dell'utente
    
    skf = StratifiedKFold(n_splits=6)
    n_classes = len(np.unique(y))
    conf_matrix_total = np.zeros((n_classes, n_classes))
    f1_macro_scores = []
    accuracies = []
    precision_scores=[]
    recall_scores=[]
    for train_index, test_index in skf.split(X_final, y):
        X_train, X_test = X_final.iloc[train_index,:], X_final.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        y_trainb, y_testb = y_bin[train_index], y_bin[test_index]
        X_train=X_train.dropna()
        indices_list = y_train.index.tolist()
        for index in empty:
            if index in indices_list:
                y_train=y_train.drop(index)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        # Alleniamo il modello
        model.fit(X_resampled, y_resampled)
        y_prob=model.predict_proba(X_test)
        # Facciamo le predizioni
        y_pred = model.predict(X_test)
    
        # Calcoliamo la matrice di confusione per questo fold
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        # Sommiamo la matrice di confusione a quella totale
        conf_matrix_total += conf_matrix
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_macro_scores.append(f1_macro)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,average='macro')
        recall = recall_score(y_test, y_pred,average='macro')
        precision_scores.append(precision)
        recall_scores.append(recall)

        accuracies.append(accuracy)
        for i in range(n_classes):
            if np.sum(y_testb[:, i]) == 0:  # Verifica se la classe 'i' è presente nel set di test
                continue  # Salta questa classe se non ci sono esempi
            fpr, tpr, _ = roc_curve(y_testb[:, i], y_prob[:, i])
            fprs[i].append(fpr)
            tprs[i].append(tpr)
            auc_value = auc(fpr, tpr)
            aucs[i].append(auc_value)

        print(f"F1 Macro per il fold corrente: {f1_macro:.4f}")
    mean_accuracy = np.mean(accuracies)
    mean_f1_macro = np.mean(f1_macro_scores)

    # Valutazione delle prestazioni
    y_user_pred = model.predict(user_final)
    y_user_prob = model.predict_proba(user_final)
    accuracy = accuracy_score(y_test, y_pred)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_accuracy = np.mean(accuracies)
    mean_f1_macro = np.mean(f1_macro_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_accuracy = np.std(accuracies)
    std_f1_macro = np.std(f1_macro_scores)
    print(f"\nF1 Macro medio su tutti i fold: {mean_f1_macro:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean precision: {mean_precision:.4f}")
    print(f"Mean recall: {mean_recall:.4f}")
    print(f"\nF1 deviazione std: {std_f1_macro:.4f}")
    print(f"\n recall deviazione std: {std_recall:.4f}")
    print(f"\naccuracy std: {std_accuracy:.4f}")
    print(f"\nprecision std: {std_precision:.4f}")
    nome_metriche=['precisione','richiamo','accuracy','f1']
    metriche_valori = [mean_precision, mean_recall, mean_accuracy, mean_f1_macro]
    deviazioni =[std_precision,std_recall,std_accuracy,std_f1_macro]
    plt.figure()
    plt.title('Grafico delle metriche del xg_boost')
    plt.bar(nome_metriche, metriche_valori, yerr=deviazioni, capsize=5,color='orange', align="center")
    plt.ylabel('Valore')
    plt.savefig("images/xg_boost/metriche.png")
    plt.close()
    class_labels = {0: 'Dead', 1: 'Seed', 2: 'Series'}
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        if len(fprs[i]) == 0:  # Verifica se ci sono dati per questa classe
            continue

        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []

        for fpr, tpr in zip(fprs[i], tprs[i]):
            all_tprs.append(np.interp(mean_fpr, fpr, tpr))

        mean_tpr = np.mean(all_tprs, axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = np.mean(aucs[i])
        plt.plot(mean_fpr, mean_tpr, label=f'{class_labels[i]} (AUC = {mean_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Curva ROC per ciascuna classe')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("images/xg_boost/auc_curve.png")
    plt.close()
    # Binarizzare le etichette per l'approccio one-vs-rest
    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    n_classes = y_test_binarized.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

   # for i in range(n_classes):
        
        #fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        #roc_auc[i] = auc(fpr[i], tpr[i])

    

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Valutazione del modello usando cross-validation
    scores = cross_val_score(model, X_final, y, cv=kf, scoring='accuracy')
    y_user_pred = y_user_pred.item()
    # Scrittura delle metriche su file
    try:
        with open('output.txt', 'a') as file:
            file.write("\nXGBoost\n")
            file.write(f"\naccuratezza: {mean_accuracy}\n")
            #file.write(f"Precisione: {precision}\n")
            #file.write(f"Richiamo: {recall}\n")
            file.write(f"F1 Score: {mean_f1_macro}\n")
            #file.write(f"Log Loss: {logloss}\n")
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
    feature_importances_nonzero = feature_importances_sorted[feature_importances_sorted > 0.007]
    feature_names_nonzero = feature_names_sorted[:len(feature_importances_nonzero)]

    # Creazione del grafico dell'importanza delle feature
    plt.figure()
    plt.title('Importanza delle Feature')
    plt.bar(range(len(feature_importances_nonzero)), feature_importances_nonzero, color="orange", align="center")
    plt.xticks(range(len(feature_names_nonzero)), feature_names_nonzero, rotation=90, fontsize=8)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig("images/xg_boost/xgb_feature.png")
    plt.close()

    conf_matrix_total = conf_matrix_total.astype(int)
    # Creazione della matrice di confusione
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_total, annot=True,fmt='d', cmap='Blues',xticklabels=stage,yticklabels=stage)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrice di Confusione')
    plt.savefig('images/xg_boost/confusion_matrix_xgb.png')
    plt.close()

    # Curva ROC
    '''
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
    '''
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
