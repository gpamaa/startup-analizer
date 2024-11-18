import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.model_selection import cross_val_score, KFold
import shap
import random
from sklearn.model_selection import StratifiedKFold
# Funzione per Random Forest
def random_forest_model(data, categorical, numeric, target, user_entries):
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

    # Creazione del modello Random Forest
    model = RandomForestClassifier(
        n_estimators=100,               # Numero di alberi
        max_depth=6,                    # Profondità massima dell'albero
        random_state=42,
        n_jobs=-1                       # Uso di tutti i core disponibili
    )

    # Addestramento del modello
    model.fit(X_train, y_train)

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
    skf = StratifiedKFold(n_splits=10)
    n_classes = len(np.unique(y))
    conf_matrix_total = np.zeros((n_classes, n_classes))
    print(len(X_final))
    print(X_final.iloc[1,:])
    f1_macro_scores = []
    accuracies = []
    for train_index, test_index in skf.split(X_final, y):
        X_train, X_test = X_final.iloc[train_index,:], X_final.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        # Alleniamo il modello
        model.fit(X_train, y_train)
    
        # Facciamo le predizioni
        y_pred = model.predict(X_test)
    
        # Calcoliamo la matrice di confusione per questo fold
        #conf_matrix = confusion_matrix(y_test, y_pred)
    
        # Sommiamo la matrice di confusione a quella totale
        #conf_matrix_total += conf_matrix
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_macro_scores.append(f1_macro)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"F1 Macro per il fold corrente: {f1_macro:.4f}")
    mean_accuracy = np.mean(accuracies)
    mean_f1_macro = np.mean(f1_macro_scores)
    print(f"\nF1 Macro medio su tutti i fold: {mean_f1_macro:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    
    
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
            file.write("\nRandom Forest\n")
            file.write(f"\naccuratezza: {scores.mean()}\n")
            file.write(f"Precisione: {precision}\n")
            file.write(f"Richiamo: {recall}\n")
            file.write(f"F1 Score: {f1}\n")
            file.write(f"Log Loss: {logloss}\n")
            file.write(f"Predizione per i dati dell'utente:" + stage[y_user_pred] + "\n")
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=stage,yticklabels=stage)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrice di Confusione')
    plt.savefig('images/random_forest/confusion_matrix_rf.png')
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
    plt.savefig('images/random_forest/roc_curve_rf.png')
    plt.close()
   
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
    return y_user_pred
