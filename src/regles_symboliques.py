"""
regles_symboliques.py

Centralise les seuils calibrés sur PLAID et les règles symboliques.

⚠️ NOTE : Les règles actuelles sont désactivées car elles dégradent
les performances sur le dataset actuel. Elles seront réactivées
quand on aura des règles plus fines (basées sur les régimes PLAID).
"""

SEUILS_PLAID = {
    "Fridge": {
        "P_steady_mean_W": 245.69,
        "P_steady_std_W": 237.35,
        "P_peak_mean_W": 619.14,
    },
    "WashingMachine": {
        "P_steady_mean_W": 262.86,
        "P_steady_std_W": 165.04,
        "P_peak_mean_W": 1028.07,
    },
    "Television": {
        "P_steady_mean_W": 80.0,
        "P_steady_std_W": 30.0,
        "P_peak_mean_W": 100.0,
    },
}

# ============================================================
# RÈGLES SYMBOLIQUES (désactivées pour l'instant)
# ============================================================

REGLES_PAR_APPAREIL = {
    # Règles désactivées tant qu'elles ne sont pas affinées.
    # "Fridge": {...},
    # "WashingMachine": {...},
}

def appliquer_regles(appliance, delta, hour, pred):
    """Applique les règles symboliques si elles existent."""
    if appliance not in REGLES_PAR_APPAREIL:
        return pred  # Pas de règle → prédiction inchangée
    regle = REGLES_PAR_APPAREIL[appliance]
    if regle["condition"](delta, hour, pred):
        return regle["boost"](delta, pred)
    return pred
