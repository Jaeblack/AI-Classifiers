import input_preprocess
from knn_classifier import KNNClassifier
from min_distance_classifier import MinimumDistanceClassifier
import random

def preprocess_data(document):
    """
    Preprocesses the data by removing unwanted characters and calculating statistics.
    """
    for content in document:
        content[-1] = content[-1].strip()
        content[0] = content[0].replace("ï»¿", "")

    attributes_number = len(document[0])
    registers_number = len(document) - 1

    print(f"Numero de registros: {registers_number}")
    print(f"Numero de atributos: {attributes_number}")
    print("______________________________________________________________")

    attributes_names = document[0]

    # Number of classes
    classes = {}
    for register in document[1:]:
        if register[-1] not in classes.keys() and register[-1] != "":
            classes[register[-1]] = 0
    print(f"Numero de clases: {len(classes.keys())}")

    # Number of registers per class
    for class_name in classes.keys():
        registers = [register for register in document if register[-1] == class_name]
        classes[class_name] = len(registers)
        print(f"\t{class_name} = {len(registers)}. ( {round(len(registers) / registers_number * 100, 2)}% )")

    # Find registers with missing values
    missing_values = []
    for i in range(1, len(document)):
        for j in range(len(document[0])):
            if document[i][j] == "":
                missing_values.append((i, j + 1, document[i]))
    print(f"Numero de tuplas con valores faltantes: {len(missing_values)}")
    for value in missing_values:
        print(f"\tRenglon: {value[0]}\tColumna: {value[1]}\tValor: {value[2]}")

    # Number of missing values per attribute and per attribute-class
    missing_values_per_attribute = {}
    for i in range(len(document[0])):
        missing_values_per_attribute[i + 1] = 0
    missing_values_per_attribute_class = {}
    for i in range(len(document[0])):
        missing_values_per_attribute_class[i + 1] = {}
        for class_name in classes.keys():
            missing_values_per_attribute_class[i + 1][class_name] = 0

    for row, col, tuple in missing_values:
        missing_values_per_attribute[col] += 1
        missing_values_per_attribute_class[col][tuple[-1]] += 1

    print("______________________________________________________________")
    print("Numero de valores faltantes por atributo:")
    # Number and percentage of missing values per attribute
    for key in missing_values_per_attribute.keys():
        print(
            f"\tAtributo {key} ({attributes_names[key - 1]}): de {registers_number} faltan {missing_values_per_attribute[key]} ({round(missing_values_per_attribute[key] / registers_number * 100, 2)}%)"
        )
        # Number and percentage of missing values per attribute-class
        for class_name in classes.keys():
            print(
                f"\t\t{class_name}: faltan {missing_values_per_attribute_class[key][class_name]} ({round(missing_values_per_attribute_class[key][class_name] / registers_number * 100, 2)}%)"
            )
        print("")

    return classes, missing_values_per_attribute, missing_values_per_attribute_class

def print_random_registers(document):
    """
    Prints 10 random registers from the document.
    """
    print("______________________________________________________________")
    print("10 registros aleatorios:")
    for i in range(10):
        print(f"\t{random.choice(document[1:])}")

def calculate_mean_and_std(document, classes):
    """
    Calculates the mean and standard deviation per attribute per class.
    """
    attributes_names = document[0]
    mean_per_attribute_per_class = {}
    std_per_attribute_per_class = {}
    for i in range(len(document[0]) - 1):
        mean_per_attribute_per_class[i + 1] = {}
        std_per_attribute_per_class[i + 1] = {}
        for class_name in classes.keys():
            mean_per_attribute_per_class[i + 1][class_name] = 0
            std_per_attribute_per_class[i + 1][class_name] = 0
    for row in document[1:]:
        for i in range(len(row) - 1):
            if row[i] != "":
                mean_per_attribute_per_class[i + 1][row[-1]] += float(row[i])
    for key in mean_per_attribute_per_class.keys():
        for class_name in classes.keys():
            mean_per_attribute_per_class[key][class_name] /= classes[class_name]
    for row in document[1:]:
        for i in range(len(row) - 1):
            if row[i] != "":
                std_per_attribute_per_class[i + 1][row[-1]] += (
                    float(row[i]) - mean_per_attribute_per_class[i + 1][row[-1]]
                ) ** 2
    for key in std_per_attribute_per_class.keys():
        for class_name in classes.keys():
            std_per_attribute_per_class[key][class_name] = (
                std_per_attribute_per_class[key][class_name] / (classes[class_name] - 1)
            ) ** 0.5
    print("______________________________________________________________")
    print("Media y desviacion estandar por atributo por clase:")
    for key in mean_per_attribute_per_class.keys():
        print(f"\tAtributo {key} ({attributes_names[key - 1]}):")
        for class_name in classes.keys():
            print(
                f"\t\t{class_name}: media = {round(mean_per_attribute_per_class[key][class_name],3)}, desviacion estandar = {round(std_per_attribute_per_class[key][class_name],3)}"
            )
        print("")

    return mean_per_attribute_per_class, std_per_attribute_per_class

def find_atipical_values(document, mean_per_attribute_per_class, std_per_attribute_per_class):
    """
    Finds tuples with atypical values.
    """
    atipical_values = []
    for i in range(1, len(document)):
        for j in range(len(document[0]) - 1):
            if document[i][j] != "":
                if (
                    abs(float(document[i][j]) - mean_per_attribute_per_class[j + 1][document[i][-1]])
                    > 2 * std_per_attribute_per_class[j + 1][document[i][-1]]
                ):
                    atipical_values.append(
                        (
                            i,
                            j + 1,
                            document[i],
                            mean_per_attribute_per_class[j + 1][document[i][-1]],
                            std_per_attribute_per_class[j + 1][document[i][-1]],
                        )
                    )
    print("______________________________________________________________")
    print("Numero de tuplas con valores atipicos: ", len(atipical_values))
    for value in atipical_values:
        print(
            f"\tR: {value[0]}, C: {value[1]}.\tValor: {value[2][value[1] - 1]}. Media: {round(value[3],3)}. std: {round(value[4],3)}. R[ {round((value[3]-(value[4]*2)),3)} - {round((value[3]+(value[4]*2)),3)} ]\tClase: {value[2][-1]}"
        )

    return atipical_values

def select_columns_to_delete(document):
    """
    Selects columns to delete from the document.
    """
    print("______________________________________________________________")
    print("Escoja las columnas a eliminar (1,2,3,4...) o escriba 0 para no eliminar ninguna:")
    columns_to_delete = input()
    columns_to_delete = columns_to_delete.split(",")
    columns_to_delete = [int(column) for column in columns_to_delete]
    new_document = []
    for row in document:
        new_row = []
        for i in range(len(row)):
            if i + 1 not in columns_to_delete:
                new_row.append(row[i])
        new_document.append(new_row)

    return new_document

def delete_rows_with_missing_values(new_document):
    """
    Deletes rows with missing values from the new document.
    """
    print("Desear eliminar las tuplas con valores faltantes? (s/n)")
    delete_rows_with_missing_values = input()
    if delete_rows_with_missing_values == "s":
        new_document = [row for row in new_document if "" not in row]
    return new_document

def delete_rows_with_atipical_values(new_document, atipical_values):
    """
    Deletes rows with atypical values from the new document.
    """
    print("Desea eliminar las tuplas con valores atipicos? (s/n)")
    delete_rows_with_atipical_values = input()
    if delete_rows_with_atipical_values == "s":
        for row, col, val, mean, std in atipical_values:
            new_document = [row for row in new_document if row != val]
    return new_document

def normalize_values(new_document):
    """
    Normalizes the values of the new document with the max of each column.
    """
    print("Desea normalizar los valores? (s/n)")
    normalize_values = input()
    if normalize_values == "s":
        # Store the maximun of each column
        max_per_column = {}
        for i in range(1,len(new_document)):
            for j in range(len(new_document[0])-1):
                if new_document[i][j] == "":
                    continue
                if j + 1 not in max_per_column.keys():
                    max_per_column[j + 1] = float(new_document[i][j])
                else:
                    if float(new_document[i][j]) > max_per_column[j + 1]:
                        max_per_column[j + 1] = float(new_document[i][j])
        # Normalize the values
        for i in range(1,len(new_document)):
            for j in range(len(new_document[0])-1):
                if new_document[i][j] == "":
                    continue
                new_document[i][j] = str(float(new_document[i][j]) / max_per_column[j + 1])
        
    return new_document

def save_new_document(new_document):
    """
    Saves the new document to a file.
    """
    new_file_name = "DB_new.csv"
    save_file = open(new_file_name, "w")
    for row in new_document:
        for i in range(len(row)):
            save_file.write(row[i])
            if i != len(row) - 1:
                save_file.write(",")
        save_file.write("\n")
    print("______________________________________________________________")
    print(f"Se ha guardado el nuevo documento en el archivo {new_file_name}")

# File and delimiter
FILE_NAME = "iris.csv"
DELIMITATOR = ","

# Read and preprocess the data
document = input_preprocess.read_from_file(FILE_NAME, DELIMITATOR)

# Preprocess the data
classes, missing_values_per_attribute, missing_values_per_attribute_class = preprocess_data(document)

# Print 10 random registers
print_random_registers(document)

# Calculate mean and standard deviation per attribute per class
mean_per_attribute_per_class, std_per_attribute_per_class = calculate_mean_and_std(document, classes)

# Find tuples with atypical values
atipical_values = find_atipical_values(document, mean_per_attribute_per_class, std_per_attribute_per_class)

# Select columns to delete
new_document = select_columns_to_delete(document)

# Delete rows with missing values
new_document = delete_rows_with_missing_values(new_document)

# Delete rows with atypical values
new_document = delete_rows_with_atipical_values(new_document, atipical_values)

# Normalize values
new_document = normalize_values(new_document)

# Save the new document
save_new_document(new_document)
