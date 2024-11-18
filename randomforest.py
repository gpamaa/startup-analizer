import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, log_loss,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.model_selection import cross_val_score, KFold
import shap
import random
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE,SMOTENC
from sklearn.preprocessing import LabelBinarizer
# Funzione per Random Forest
def random_forest_model(data, categorical, numeric, target, user_entries):
    # Separazione delle feature e del target
    X = data[categorical + numeric]
    y = data[target]
    stage=['dead','seed','series']
    empty= [134,142,143,154]

    # Codifica delle feature categoriche
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(data[categorical])
   
    # Creazione del DataFrame codificato
    encoded_columns = encoder.get_feature_names_out(categorical)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)
    
    # Concatenazione delle feature codificate con quelle numeriche
    X_final = pd.concat([X_encoded_df, data[numeric]], axis=1)
    
    # Suddivisione del dataset in training e test set
    
    # Creazione del modello Random Forest
    model = RandomForestClassifier(
        n_estimators=100,               # Numero di alberi
        max_depth=6,                    # Profondità massima dell'albero
        random_state=42,
        n_jobs=-1                       # Uso di tutti i core disponibili
    )

    ros = RandomOverSampler()
    #X_resampled, y_resampled = ros.fit_resample(X_final, y)
    smote = SMOTE()
    #X_resampled, y_resampled = smote.fit_resample(X_final, y)
    # Codifica dei dati di input dell'utente
    user_encoded = encoder.transform(user_entries[categorical])
    user_encoded_df = pd.DataFrame(user_encoded, columns=encoded_columns)
    
    # Concatenazione dei dati di input codificati con quelli numerici
    user_final = pd.concat([user_encoded_df, user_entries[numeric]], axis=1)
    
    # Predizione del target per i dati di input dell'utente
   
    skf = StratifiedKFold(n_splits=10)
    n_classes = len(np.unique(y))
    conf_matrix_total = np.zeros((n_classes, n_classes))
    f1_macro_scores = []
    accuracies = []
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    tprs = {i: [] for i in range(n_classes)}
    fprs = {i: [] for i in range(n_classes)}
    aucs = {i: [] for i in range(n_classes)}
    # Liste per raccogliere le metriche
    precision_scores = []
    recall_scores = []
    roc_auc_scores = []
    y_true_all = []
    y_scores_all = []
    for train_index, test_index in skf.split(X_final, y):
        X_train, X_test = X_final.iloc[train_index,:], X_final.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        y_trainb, y_testb = y_bin[train_index], y_bin[test_index]
        X_train=X_train.dropna()
        indices_list = y_train.index.tolist()
        for index in empty:
            if index in indices_list:
                y_train=y_train.drop(index)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        # Alleniamo il modello
        model.fit(X_resampled, y_resampled)

        # Facciamo le predizioni
        y_pred = model.predict(X_test)
        
        y_scores = model.predict_proba(X_test)
        print("cuai fancul")
        print(y_test.shape)
    # Calcola le metriche
        precision = precision_score(y_test, y_pred,average='macro')
        recall = recall_score(y_test, y_pred,average='macro')
        for i in range(n_classes):
            if np.sum(y_testb[:, i]) == 0:  # Verifica se la classe 'i' è presente nel set di test
                continue  # Salta questa classe se non ci sono esempi

            fpr, tpr, _ = roc_curve(y_testb[:, i], y_scores[:, i])
            fprs[i].append(fpr)
            tprs[i].append(tpr)
            auc_value = auc(fpr, tpr)
            aucs[i].append(auc_value)
    
    # Salva le metriche
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Accumula i veri valori e i punteggi per la curva ROC
        y_true_all.extend(y_test)
        y_scores_all.extend(y_scores)

# Calcola la deviazione standard delle metriche

        # Calcoliamo la matrice di confusione per questo fold
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        # Sommiamo la matrice di confusione a quella totale
        conf_matrix_total += conf_matrix
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_macro_scores.append(f1_macro)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"F1 Macro per il fold corrente: {f1_macro:.4f}")
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_roc_auc = np.mean(roc_auc_scores)
    mean_accuracy = np.mean(accuracies)
    mean_f1_macro = np.mean(f1_macro_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_roc_auc = np.std(roc_auc_scores)
    std_accuracy = np.std(accuracies)
    std_f1_macro = np.std(f1_macro_scores)
    nome_metriche=['precisione','richiamo','accuracy','f1']
    metriche_valori = [mean_precision, mean_recall, mean_accuracy, mean_f1_macro]
    deviazioni =[std_precision,std_recall,std_accuracy,std_f1_macro]
    folds = range(1, len(roc_auc_scores) + 1)
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

    plt.savefig("images/random_forest/auc_curve.png")
    plt.close()
    print(f"\nF1 Macro medio su tutti i fold: {mean_f1_macro:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"precision {mean_precision:.4f}")
    print(f"recall {mean_recall:.4f}")
    print(f"\nF1 deviazione std: {std_f1_macro:.4f}")
    print(f"\n recall deviazione std: {std_recall:.4f}")
    print(f"\naccuracy std: {std_accuracy:.4f}")
    print(f"\nprecision std: {std_precision:.4f}")
    y_user_pred = model.predict(user_final)
    y_user_prob = model.predict_proba(user_final)
    print(y_user_pred)
    plt.figure()
    plt.title('Grafico delle metriche del random forest')
    plt.bar(nome_metriche, metriche_valori, yerr=deviazioni, capsize=5,color='orange', align="center")
    plt.ylabel('Valore')
    plt.savefig("images/random_forest/metriche.png")
    plt.close()



    # Riordinamento delle feature in base all'importanza
    feature_importances = model.feature_importances_
    feature_names = X_final.columns
    sorted_indices = np.argsort(feature_importances)[::-1]
    feature_importances_sorted = feature_importances[sorted_indices]
    feature_names_sorted = feature_names[sorted_indices]

    # Filtraggio delle feature con importanza non nulla
    feature_importances_nonzero = feature_importances_sorted[feature_importances_sorted > 0]
    feature_names_nonzero = feature_names_sorted[:len(feature_importances_nonzero)]
    feature_names_nonzero = feature_names_nonzero[:25]
    feature_importances_nonzero= feature_importances_nonzero[:25]
    # Creazione del grafico dell'importanza delle feature
    plt.figure()
    plt.title('Importanza delle Feature')
    plt.bar(range(len(feature_importances_nonzero)), feature_importances_nonzero, color="orange", align="center")
    plt.xticks(range(len(feature_names_nonzero)), feature_names_nonzero, rotation=90, fontsize=8)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig("images/random_forest/rf_feature.png")
    plt.close()

    # Creazione della matrice di confusione
    conf_matrix_total=conf_matrix_total.astype(int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_total, annot=True, fmt='d', cmap='Blues',xticklabels=stage,yticklabels=stage)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrice di Confusione')
    plt.savefig('images/random_forest/confusion_matrix_rf.png')
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
    plt.savefig('images/random_forest/roc_curve_rf.png')
    plt.close()
    '''
    # SHAP plot
    numero_casuale = random.randint(0, 2)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_final)
    for i in range(3):
        plt.figure()
        shap.plots.waterfall(shap_values[numero_casuale, :, i], show=False)
        plt.title(f"grafico per il target " + stage[i])
        plt.savefig(f'images/random_forest/shap/waterfall_plot/' + stage[i] + '.png', bbox_inches='tight')
        plt.close()
        plt.figure()
        shap.summary_plot(shap_values[:,:,i], X_final, show=False)
        plt.title(f"grafico per il target " + stage[i])
        plt.savefig(f'images/random_forest/shap/summary_plot/' + stage[i] + '.png', bbox_inches='tight')
        plt.close()
        plt.figure()
        shap.plots.bar(shap_values[:,:,i], show=False)
        plt.title(f"grafico per il target " + stage[i])
        plt.savefig(f'images/random_forest/shap/barplot/' + stage[i] + '.png', bbox_inches='tight')
        plt.close()
        
    return y_user_pred[0]
