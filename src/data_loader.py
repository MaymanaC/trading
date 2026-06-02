"""Chargement des données depuis le fichier Excel."""
import pandas as pd
import numpy as np


def load_crypto_data(filepath: str) -> pd.DataFrame:
    """
    Charge les prix journaliers Bitcoin / Ethereum.
    Retourne un DataFrame avec colonnes ['Bitcoin', 'Ethereum'], indexé par Date.
    """
    df = pd.read_excel(filepath, sheet_name='Cryptos', index_col=0, header=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df = df.iloc[:, :2].copy()
    df.columns = ['Bitcoin', 'Ethereum']
    df = df.dropna(subset=['Bitcoin', 'Ethereum'])
    df = df.sort_index()
    return df


def load_cds_data(filepath: str) -> pd.DataFrame:
    """
    Charge les spreads journaliers CDX 5Y / ITX 5Y.
    Convertit les spreads en prix CDS : CDSPrice = Spread * 5 (nominal 100, maturité 5ans).
    """
    df = pd.read_excel(filepath, sheet_name='CDS Indices', index_col=0, header=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df = df.iloc[:, :2].copy()
    df.columns = ['CDX_spread', 'ITX_spread']
    df = df.dropna(subset=['CDX_spread', 'ITX_spread'])
    df['CDX'] = df['CDX_spread'] * 5
    df['ITX'] = df['ITX_spread'] * 5
    df = df.sort_index()
    return df
