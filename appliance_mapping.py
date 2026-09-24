# ============================================================
# appliance_mapping.py
# Mapping officiel REFIT House 1, confirmé depuis
# data/raw/REFIT_meta/MetaData_Tables.xlsx (feuille "House 1").
# Source unique de vérité — tous les scripts du projet doivent
# importer ce fichier plutôt que redéfinir leur propre mapping.
# ============================================================

REFIT_HOUSE1_APPLIANCES = {
    "Appliance1": {"name": "Fridge", "brand": "Hotpoint RLA50P"},
    "Appliance2": {"name": "Freezer1", "brand": "Beko CF393APW"},
    "Appliance3": {"name": "Freezer2", "brand": "Unknown"},
    "Appliance4": {"name": "WasherDryer", "brand": "Creda T522VW"},
    "Appliance5": {"name": "WashingMachine", "brand": "Beko WMC6140"},
    "Appliance6": {"name": "Dishwasher", "brand": "Bosch"},
    "Appliance7": {"name": "Computer", "brand": "Lenovo H520s"},
    "Appliance8": {"name": "Television", "brand": "Toshiba 32BL502b"},
    "Appliance9": {"name": "ElectricHeater", "brand": "GLEN 2172"},
}

# Appareils communs entre REFIT House 1 et iAWE (par nom sémantique,
# pas par numéro de canal) — c'est CE mapping qu'il faut utiliser
# pour la fusion, pas les anciens numéros de canal supposés au
# début de la session.
REFIT_TO_IAWE_COMMON = {
    "Appliance1": "Fridge",           # REFIT Fridge <-> iAWE Fridge
    "Appliance5": "WashingMachine",   # REFIT WashingMachine <-> iAWE WashingMachine
    "Appliance8": "Television",       # REFIT Television <-> iAWE Television
}

# Appareils REFIT sans équivalent iAWE — restent sur leur source
# unique, jamais fusionnés avec une autre maison.
REFIT_EXCLUSIVE = ["Appliance2", "Appliance3", "Appliance4", "Appliance6", "Appliance7", "Appliance9"]

# Appareils iAWE sans équivalent REFIT — idem.
IAWE_EXCLUSIVE = ["AC_Unit1", "AC_Unit2", "ClothesIron", "WaterMotor"]


def get_appliance_name(refit_column):
    """Renvoie le nom réel de l'appareil pour une colonne AppplianceN."""
    return REFIT_HOUSE1_APPLIANCES.get(refit_column, {}).get("name", refit_column)


if __name__ == "__main__":
    print("Mapping REFIT House 1 (officiel, vérifié) :")
    for col, info in REFIT_HOUSE1_APPLIANCES.items():
        marker = " ← FUSIONNÉ avec iAWE" if col in REFIT_TO_IAWE_COMMON else ""
        print(f"  {col} = {info['name']} ({info['brand']}){marker}")
