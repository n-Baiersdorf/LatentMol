# normalized_constants.py

# Dictionary containing normalized atomic properties for molecular modeling
SPV = 0.0  # Normalized semantic padding value

def normalize_value(value, normalization_type, column_index):
    if value == -1:
        return 0.0
        
    if normalization_type == 1:  # Simple scaling
        max_values = [53, 126.904, 140, 139, 198, 3.98, 1681.0, 3388.3, 6050.4, 349.0, 7, 5]
        return (value / max_values[column_index]) * 0.5
        
    elif normalization_type == 2:  # Min-max normalization
        min_values = [1, 1.008, 50, 31, 120, 2.19, 941.0, 1845.9, 2914.1, -7.0, 1, -3]
        max_values = [53, 126.904, 140, 139, 198, 3.98, 1681.0, 3388.3, 6050.4, 349.0, 7, 5]
        return ((value - min_values[column_index]) / 
                (max_values[column_index] - min_values[column_index])) * 0.5
    return 0.0

# Normalization configuration
# 1: Simple scaling
# 2: Min-max normalization
normalization_config = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2]

# Pre-computed normalized dictionary
atom_dict_normalized = {
    element: [normalize_value(val, normalization_config[i], i) 
             for i, val in enumerate(values)]
    for element, values in {
        "H": [ 1, 1.008, 53, 31, 120, 2.20, 1312.0, -1, -1, 72.8, 1, 1],
        "C": [ 6, 12.011, 70, 77, 170, 2.55, 1086.5, 2352.6, 4620.5, 121.9, 4, 4],
        "N": [ 7, 14.007, 65, 71, 155, 3.04, 1402.3, 2856.0, 4578.1, -7.0, 5, -3],
        "O": [ 8, 15.999, 60, 66, 152, 3.44, 1313.9, 3388.3, 5300.5, 141.0, 6, -2],
        "F": [ 9, 18.998, 50, 57, 147, 3.98, 1681.0, 3374.2, 6050.4, 328.0, 7, -1],
        "P": [15, 30.974, 100, 107, 180, 2.19, 1011.8, 1907.0, 2914.1, 72.0, 5, 5],
        "S": [16, 32.065, 100, 105, 180, 2.58, 999.6, 2252.0, 3357.0, 200.0, 6, -2],
        "Cl": [17, 35.453, 100, 102, 175, 3.16, 1251.2, 2298.0, 3822.0, 349.0, 7, -1],
        "Br": [35, 79.904, 115, 120, 185, 2.96, 1139.9, 2103.0, 3470.0, 324.6, 7, -1],
        "I": [53, 126.904, 140, 139, 198, 2.66, 1008.4, 1845.9, 3180.0, 295.2, 7, -1],
        "Se": [34, 78.96, 115, 120, 190, 2.55, 941.0, 2045.0, 2973.7, 195.0, 6, -2]
    }.items()
}


# Units and Descriptions:
# OZ: Atomic Number (dimensionless)
# AM: Atomic Mass (u, unified atomic mass units)
# AR: Atomic Radius (pm, picometers)
# KR: Covalent Radius (pm, picometers)
# VWR: Van der Waals Radius (pm, picometers)
# EN: Electronegativity (Pauling scale, dimensionless)
# IE1: First Ionization Energy (kJ/mol)
# IE2: Second Ionization Energy (kJ/mol)
# IE3: Third Ionization Energy (kJ/mol)
# EA: Electron Affinity (kJ/mol)
# VAL: Valence Electrons (dimensionless)
# OX: Common Oxidation State (dimensionless)
