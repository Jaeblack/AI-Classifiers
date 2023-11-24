
import json

FILE_NAME = str(input("Nombre del archivo: ")) # Ejemplo: "maternal.csv"
DELIMITATOR = str(input("Delimitador: ")) # Ejemplo: ","


def read_from_file(filename):
    with open(filename, 'r') as file:
        return [line.split(DELIMITATOR) for line in file.readlines()]
    
def label_attributes(document, attributes_number, patterns_number):
    attributes = []
    for i in range(attributes_number):
        data_type = ""
        is_float = False
        is_number = True
        for j in range(1, patterns_number):
            if "." in document[j][i]:
                is_float = True
            if not document[j][i].isdigit() or not document[j][i].isnumeric():
                is_number = False
            
        if is_float == True:
            data_type = "real. rango: ["+ str(min([float(x[i]) for x in document[1:]])) + " - " + str(max([float(x[i]) for x in document[1:]])) + "]"
        elif is_number == True:
            if len(set([int(x[i]) for x in document[1:]])) == 2:
                data_type = "binario. valores: " + str(set([int(x[i]) for x in document[1:]]))
            else:
                data_type = "entero. rango: ["+ str(min([int(x[i]) for x in document[1:]])) + " - " + str(max([int(x[i]) for x in document[1:]])) + "]"
        else:
            data_type = "categorico. valores: " + str(set([x[i] for x in document[1:]]))
        attributes.append({"nombre": document[0][i], "tipo": data_type})
    return attributes

def default_selection(attributes):
    attributes_input = []
    for attribute in attributes:
        if "categorico" not in attribute["tipo"]:
            attributes_input.append(attribute)

    attributes_output = []
    for attribute in attributes:
        if "categorico" in attribute["tipo"]:
            attributes_output.append(attribute)

    return attributes_input, attributes_output

def print_attr_data(attributes, attributes_input, attributes_output, attributes_number, patterns_number):
    print(f"Numero de atributos: {attributes_number}\nNumero de patrones: {patterns_number}")
    print("Atributos:")
    for attribute in attributes:
        print(f"\t{attribute['nombre']}:\t{attribute['tipo']}")
    print("Atributos de entrada:")
    for attribute in attributes_input:
        print(f"\t{attribute['nombre']}:\t{attribute['tipo']}")
    print("Atributos de salida:")
    for attribute in attributes_output:
        print(f"\t{attribute['nombre']}:\t{attribute['tipo']}")

def make_subset(document, attributes_number, patterns_number, selection_input, selection_output, selection_patterns):
    new_matrix = []
    for i in range(1,patterns_number):
        if str(i) in selection_patterns:
            new_matrix.append([])
            inputs = []
            for j in range(attributes_number):
                if str(j+1) in selection_input:
                    inputs.append(document[i][j])
            outputs = []
            for j in range(attributes_number):
                if str(j+1) in selection_output:
                    outputs.append(document[i][j])
            new_matrix[-1].append(inputs)
            new_matrix[-1].append(outputs)
    return new_matrix

def select_subset(document, attributes_number, patterns_number):
    selection_input = str(input(f"Atributos de entrada(Numeros del 1 al {attributes_number} separados por '{DELIMITATOR}'): ")).split(DELIMITATOR)
    selection_output = str(input(f"Atributos de salida(Numeros del 1 al {attributes_number} separados por '{DELIMITATOR}'): ")).split(DELIMITATOR)
    selection_patterns = str(input(f"Patrones(Numeros del 1 al {patterns_number} separados por '{DELIMITATOR}'): ")).split(DELIMITATOR)
    return make_subset(document, attributes_number, patterns_number, selection_input, selection_output, selection_patterns)
    
def save_subset(new_matrix):
    with open('data.json', 'w') as outfile:
        json.dump(new_matrix, outfile)
    print("Matriz de datos guardada en 'data.json':")
    for i in range(len(new_matrix)):
        print(f"\t{str(i+1)}: {new_matrix[i]}")
    
document = read_from_file(FILE_NAME)
for content in document:
    #Borrar el caracter de nueva linea
    content[-1] = content[-1].strip()
    #Borrando el inicio del archivo
    content[0] = content[0].replace("ï»¿", "")

attributes_number = len(document[0])
patterns_number = len(document)
attributes = label_attributes(document, attributes_number, patterns_number)
attributes_input, attributes_output = default_selection(attributes)
print_attr_data(attributes, attributes_input, attributes_output, attributes_number, patterns_number)

new_matrix = select_subset(document, attributes_number, patterns_number)

save_subset(new_matrix)

