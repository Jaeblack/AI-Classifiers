import json

def read_from_file(filename, delimiter):
    """
    Reads data from a CSV file and returns it as a list of lists.
    """
    with open(filename, 'r') as file:
        return [line.split(delimiter) for line in file.readlines()]

def label_attributes(document, attributes_number, patterns_number):
    """
    Labels each attribute in the document based on its type (real, binary, integer, or categorical).
    """
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
            
        if is_float:
            data_type = "real. rango: [" + str(min([float(x[i]) for x in document[1:]])) + " - " + str(max([float(x[i]) for x in document[1:]])) + "]"
        elif is_number:
            if len(set([int(x[i]) for x in document[1:]])) == 2:
                data_type = "binario. valores: " + str(set([int(x[i]) for x in document[1:]]))
            else:
                data_type = "entero. rango: [" + str(min([int(x[i]) for x in document[1:]])) + " - " + str(max([int(x[i]) for x in document[1:]])) + "]"
        else:
            data_type = "categorico. valores: " + str(set([x[i] for x in document[1:]]))
        attributes.append({"nombre": document[0][i], "tipo": data_type})
    return attributes

def default_selection(attributes):
    """
    Performs a default selection of input and output attributes based on their types.
    """
    attributes_input = [attribute for attribute in attributes if "categorico" not in attribute["tipo"]]
    attributes_output = [attribute for attribute in attributes if "categorico" in attribute["tipo"]]
    return attributes_input, attributes_output

def print_attr_data(attributes, attributes_input, attributes_output, attributes_number, patterns_number):
    """
    Prints details about the attributes, including their names and types.
    """
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
    """
    Creates a subset of the document based on the selected attributes and patterns.
    """
    new_matrix = []
    for i in range(1, patterns_number):
        if str(i) in selection_patterns:
            new_matrix.append([])
            inputs = [document[i][j] for j in range(attributes_number) if str(j + 1) in selection_input]
            outputs = [document[i][j] for j in range(attributes_number) if str(j + 1) in selection_output]
            new_matrix[-1].append(inputs)
            new_matrix[-1].append(outputs)
    return new_matrix

def select_subset(document, attributes_number, patterns_number, delimiter):
    """
    Allows the user to select specific input and output attributes and patterns.
    """
    selection_input = input(f"Atributos de entrada (Numeros del 1 al {attributes_number} separados por '{delimiter}'): ").split(delimiter)
    selection_output = input(f"Atributos de salida (Numeros del 1 al {attributes_number} separados por '{delimiter}'): ").split(delimiter)

    selection_patterns_range = input(f"Patrones (Rango, por ejemplo, 1-1000): ")
    start, end = map(int, selection_patterns_range.split('-'))
    selection_patterns = [str(i) for i in range(start, end + 1)]

    return make_subset(document, attributes_number, patterns_number, selection_input, selection_output, selection_patterns)

def save_subset(new_matrix):
    """
    Saves the selected data subset into a JSON file.
    """
    with open('data.json', 'w') as outfile:
        json.dump(new_matrix, outfile)
    print("Matriz de datos guardada en 'data.json':")
    for i in range(len(new_matrix)):
        print(f"\t{str(i + 1)}: {new_matrix[i]}")
