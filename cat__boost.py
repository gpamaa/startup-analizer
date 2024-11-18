import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import KFold
import random
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE,SMOTENC
def catboost(data, categorical, numeric, target, entries):
    # Separazione delle feature e del target
    X = data[categorical + numeric]
    y = data[target]
    cat_features = [X.columns.get_loc(col) for col in categorical]
    stage=['dead','seed','series']

    # Suddivisione del dataset in training e test set
     # Codifica delle feature categoriche
    # random oversampling
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X, y)
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    tprs = {i: [] for i in range(n_classes)}
    fprs = {i: [] for i in range(n_classes)}
    aucs = {i: [] for i in range(n_classes)}
    #smote
    #X=X.dropna()
    #y=y.drop([134,142,143,154])
    #smote = SMOTENC(categorical_features=[len(categorical)], random_state=42)
    #X_resampled, y_resampled = smote.fit_resample(X, y)
   
    # Creazione del modello
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        cat_features=categorical,
        verbose=100
    )
    skf = StratifiedKFold(n_splits=10)#ricorda di settare a 10
    n_classes = len(np.unique(y))
    conf_matrix_total = np.zeros((n_classes, n_classes))
    f1_macro_scores = []
    accuracies = []
    precision_scores=[]
    recall_scores=[]
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        y_trainb, y_testb = y_bin[train_index], y_bin[test_index]
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        # Alleniamo il modello
        model.fit(X_resampled, y_resampled)
    
        # Facciamo le predizioni
        y_pred = model.predict(X_test)
        y_prob=model.predict_proba(X_test)
    
        # Calcoliamo la matrice di confusione per questo fold
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    
        # Sommiamo la matrice di confusione a quella totale
        conf_matrix_total += conf_matrix
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_macro_scores.append(f1_macro)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        precision = precision_score(y_test, y_pred,average='macro')
        recall = recall_score(y_test, y_pred,average='macro')
        precision_scores.append(precision)
        recall_scores.append(recall)
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
    class_labels = {0: 'Dead', 1: 'Seed', 2: 'Series'}
    mean_recall = np.mean(recall_scores)
    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_accuracy = np.std(accuracies)
    std_f1_macro = np.std(f1_macro_scores)
    # Predizioni sul test set
    y_pred2 = model.predict(entries)
    y_prob2 = model.predict_proba(entries)
    print(f"\nF1 Macro medio su tutti i fold: {mean_f1_macro:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"precision {mean_precision:.4f}")
    print(f"recall {mean_recall:.4f}")
    print(f"\nF1 deviazione std: {std_f1_macro:.4f}")
    print(f"\n recall deviazione std: {std_recall:.4f}")
    print(f"\naccuracy std: {std_accuracy:.4f}")
    print(f"\nprecision std: {std_precision:.4f}")
    nome_metriche=['precisione','richiamo','accuracy','f1']
    metriche_valori = [mean_precision, mean_recall, mean_accuracy, mean_f1_macro]
    deviazioni =[std_precision,std_recall,std_accuracy,std_f1_macro]
    plt.figure()
    plt.title('Grafico delle metriche del cat boost')
    plt.bar(nome_metriche, metriche_valori, yerr=deviazioni, capsize=5,color='orange', align="center")
    plt.ylabel('Valore')
    plt.savefig("images/cat_boost/metriche.png")
    plt.close()
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
    plt.savefig("images/cat_boost/auc_curve.png")
    plt.close()
    
    # Visualizzazione dell'importanza delle feature
    feature_importances = model.get_feature_importance()
    feature_importances_normalized = feature_importances / feature_importances.sum()
    
    # Creazione di un DataFrame per l'importanza delle caratteristiche
    feature_importance_df = pd.DataFrame({
        'feature': categorical + numeric,
        'importance': feature_importances_normalized
    }).sort_values(by='importance', ascending=False)
    feature_importances_normalized = feature_importance_df['importance']
    feature_names = feature_importance_df['feature']

    # Visualizzazione dell'importanza delle caratteristiche
    plt.figure()
    plt.bar(range(X.shape[1]), feature_importances_normalized, color="orange", align="center")
    plt.xticks(range(X.shape[1]), feature_names, rotation=90, fontsize=8)
    plt.subplots_adjust(bottom=0.4)
    plt.title('Importanza delle feature')
    plt.savefig('images/cat_boost/catb_feature.png')
    plt.close()
    
    # Creazione della matrice di confusione
    conf_matrix_total=conf_matrix_total.astype(int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_total, annot=True, fmt='d', cmap='Blues',xticklabels=stage,yticklabels=stage)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrice di Confusione')
    plt.savefig('images/cat_boost/catboost_matrix.png')
    plt.close()
    

    # SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    numero_casuale = random.randint(0, 9)
    
    for i in range(3):
        plt.figure()
        shap.plots.waterfall(shap_values[numero_casuale, :, i],show=False)
        plt.title(f"grafico per il target "+stage[i])
        plt.savefig(f'images/cat_boost/shap/waterfall_plot/'+stage[i]+'.png', bbox_inches='tight')
        plt.close()
        plt.figure()
        shap.summary_plot(shap_values[:,:,i], X,show=False)
        plt.title(f"grafico per il target "+stage[i])
        plt.savefig(f'images/cat_boost/shap/summary_plot/'+stage[i]+'.png', bbox_inches='tight')
        plt.close()
        plt.figure()
        shap.plots.bar(shap_values[:,:,i],show=False)
        plt.title(f"grafico per il target "+stage[i])
        plt.savefig(f'images/cat_boost/shap/barplot/'+stage[i]+'.png', bbox_inches='tight')
        plt.close()
    return y_pred2[0]
