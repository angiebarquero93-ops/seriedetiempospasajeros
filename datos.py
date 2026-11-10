import pandas as pd
import os

def cargar_datos():
    """
    Función para cargar los datos mensuales de pasajeros de avión.
    Intenta descargarlos de internet y, si no hay conexión, usa el archivo local.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    ruta_local = "datos/airline-passengers.csv"
    
    try:
        # Intentamos descargar de internet
        df = pd.read_csv(url)
    except Exception:
        # Si no hay internet, leemos el archivo local
        if os.path.exists(ruta_local):
            df = pd.read_csv(ruta_local)
        else:
            df = pd.read_csv("airline-passengers.csv")
            
    # Limpiamos nombres de columnas por si acaso
    df.columns = [c.strip() for c in df.columns]
    
    return df
