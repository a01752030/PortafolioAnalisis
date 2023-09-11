import pandas as pd

def clean(name):
    df_train = pd.read_csv(name, encoding='unicode_escape', engine='python')
    
    df_train.drop(['HomePlanet','Cabin', 'Destination', 'Name'], axis=1, inplace=True)
    
    #df_train.dropna(inplace=True)
    
    mappingbin = {'False': 0, 'True': 1}
    df_train.replace({'CryoSleep': mappingbin, 'VIP': mappingbin}, inplace=True)
    
    return df_train

if __name__ == '__main__':
    clean("train.csv")