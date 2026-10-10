







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





REFIT_TO_IAWE_COMMON = {
    "Appliance1": "Fridge",
    "Appliance5": "WashingMachine",
    "Appliance8": "Television",
}



REFIT_EXCLUSIVE = ["Appliance2", "Appliance3", "Appliance4", "Appliance6", "Appliance7", "Appliance9"]


IAWE_EXCLUSIVE = ["AC_Unit1", "AC_Unit2", "ClothesIron", "WaterMotor"]


def get_appliance_name(refit_column):
    """Renvoie le nom réel de l'appareil pour une colonne AppplianceN."""
    return REFIT_HOUSE1_APPLIANCES.get(refit_column, {}).get("name", refit_column)


if __name__ == "__main__":
    print("Mapping REFIT House 1 (officiel, vérifié) :")
    for col, info in REFIT_HOUSE1_APPLIANCES.items():
        marker = " ← FUSIONNÉ avec iAWE" if col in REFIT_TO_IAWE_COMMON else ""
        print(f"  {col} = {info['name']} ({info['brand']}){marker}")
