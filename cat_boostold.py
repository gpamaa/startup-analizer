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
from sklearn.model_selection import StratifiedKFold
def catboost(data, categorical, numeric, target, entries):
    # Separazione delle feature e del target
    X = data[categorical + numeric]
    y = data[target]
    cat_features = [X.columns.get_loc(col) for col in categorical]
    stage=['dead','seed','series']

    # Suddivisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Creazione del Pool per il training
    train_pool = Pool(X_train, y_train, cat_features=categorical)
    test_pool = Pool(X_test, y_test, cat_features=categorical)

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
    skf = StratifiedKFold(n_splits=10)
    n_classes = len(np.unique(y))
    conf_matrix_total = np.zeros((n_classes, n_classes))
    f1_macro_scores = []
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
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
    


    # KFold cross-validation
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]

        train_pool_kf = Pool(X_train_kf, y_train_kf, cat_features=cat_features)
        test_pool_kf = Pool(X_test_kf, y_test_kf, cat_features=cat_features)

        model.fit(train_pool_kf)
        scores.append(model.score(test_pool_kf))
    scores = np.array(scores)
    
    # Predizioni sul test set
    y_pred = model.predict(X_test)
    y_pred2 = model.predict(entries)
    y_prob2 = model.predict_proba(entries)


    y_prob = model.predict_proba(X_test)

    # Valutazione delle prestazioni
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    logloss = log_loss(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # ROC AUC per multiclasse
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    y_pred2= y_pred2.item()
    with open('output.txt', 'a') as file:
        file.write("\n Cat boost\n")
        file.write(f"\naccuratezza: {scores.mean()}\n")
        file.write(f"Precisione: {precision}\n")
        file.write(f"Richiamo: {recall}\n")
        file.write(f"F1 Score: {f1}\n")
        file.write(f"Log Loss: {logloss}\n")
        file.write(f"Predizione per i dati dell'utente: {stage[y_pred2]}\n")
        file.write(f"Probabilita' per i dati dell'utente: {y_prob2}\n")
    
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
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=stage,yticklabels=stage)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Matrice di Confusione')
    plt.savefig('images/cat_boost/catboost_matrix.png')
    plt.close()
    
    # Curva ROC per multiclasse
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(np.unique(y_test))):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_prob[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8, 6))
    for i in range(len(np.unique(y_test))):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve class {stage[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('images/cat_boost/roc_curve_catboost.png')
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
    return y_pred2
