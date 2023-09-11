from clean import clean
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import graphviz
import matplotlib


def DTree(x_train, y_train, x_val):
    model_t = DecisionTreeClassifier(random_state=10, criterion="entropy")
    model_t.fit(x_train, y_train)
    y_hat_t = model_t.predict(x_val)
    return y_hat_t
 
def DTreeBetter(x_train, y_train, x_val):
    model_t = DecisionTreeClassifier(max_depth=10,max_features=None,max_leaf_nodes=30,min_samples_leaf=1,min_samples_split=2, criterion="entropy")
    model_t.fit(x_train, y_train)
    y_hat_t = model_t.predict(x_val)
    return y_hat_t


def graficar(x_train, y_train):
    model_t = DecisionTreeClassifier(random_state=10, criterion="entropy")
    model_t.fit(x_train, y_train)
    myTreeData = sklearn.tree.export_graphviz(model_t)
    graphData = graphviz.Source(myTreeData)
    graphData.render("data.gv")


def concatenar(ids, emission, name):
    final = pd.DataFrame()
    final['PassengerId'] = ids
    final['Transported'] = pd.Series(emission)
    final.to_csv(name, index=False)
    print(f"checa tu csv con nombre: {name}")
    return final


if __name__ == '__main__':
    df = clean('train.csv')
    dfTest = clean('test.csv')
    
    x = df.drop(['PassengerId', 'Transported'], axis=1)
    x2 = dfTest.drop(['PassengerId'],axis=1)
    y = df['Transported']
    
    # Separar datos en entrenamiento, validación y prueba
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    # Normalización
    scaler = MinMaxScaler()
    x_train_transformada_full = scaler.fit_transform(x)
    x_train_transformada = scaler.fit_transform(x_train)
    x_val_transformada = scaler.transform(x_val)
    x_test_transformada = scaler.transform(x2)
    
    # Entrenar y evaluar
    y_train_pred = DTree(x_train_transformada, y_train, x_train_transformada)
    y_val_pred = DTree(x_train_transformada, y_train, x_val_transformada)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Accuracy en el conjunto de entrenamiento: {train_accuracy}")
    print(f"Accuracy en el conjunto de validación: {val_accuracy}")
    
    # Diagnóstico del modelo
    if train_accuracy > 0.9 and val_accuracy < 0.7:
        print("El modelo tiene un alto sesgo (underfitting).")
    elif train_accuracy > 0.9 and (val_accuracy >= 0.7 and val_accuracy <= 0.85):
        print("El modelo tiene un sesgo medio y varianza media.")
    elif train_accuracy > 0.9 and val_accuracy > 0.85:
        print("El modelo está bien ajustado (fit).")
    else:
        print("El modelo tiene alta varianza (overfitting).")
    
    # Otras funciones (graficar, generar CSV, etc.)
    # graficar(x_train_transformada, y_train)
    # concatenar(dfTest.PassengerId, y_val_pred, 'DT.csv')

    # Definir los hiperparámetros y sus posibles valores
    param_grid = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 30]
    }

# Crear modelo base
    #base_model = DecisionTreeClassifier(random_state=10, criterion="entropy")

# Instanciar GridSearchCV
    #grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Ajustar modelo a datos
    #grid_search.fit(x_train_transformada, y_train)

# Ver mejores hiperparámetros
    #print("Mejores hiperparámetros:", grid_search.best_params_)

    DT = DTreeBetter(x_train_transformada_full, y, x_test_transformada)
    concatenar(dfTest['PassengerId'], DT, 'DT.csv')