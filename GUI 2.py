#This code is for a GUI to select and retrieve cases
import customtkinter as ctk
import pandas as pd
import numpy as np

import Methods2
from Methods2 import CaseBase,Diversity,ModifiedCNN,SearchSimilarModCNN, DescriptionsAndSolutions

# Initialize the app
app = ctk.CTk()
app.title("Input Variables and Variable Weights")
app.geometry("600x300")

# Create a frame for the inputs and weights
frame = ctk.CTkFrame(app)
frame.pack(pady=100, padx=200)

# Define labels for input variables
labels_input = [
    "Task:", "Case study type:", "Case study:", "Online/Off-line:", "Input for the model:"
]

# Define dropdown options for each input
dropdown_options = [
    ["Health modelling","Health assessment","Remaining useful life estimation","One step future state forecast","Multiple steps future state forecast","Fault identification","Fault feature extraction","Fault detection"],
    ["Rotary machines","Structures","Production lines","Reciprocating machines","Electrical components","Lubricants","Electromechanical systems","Optical devices","Energy cells and batteries","Unknown Item Type","Pipelines and ducts","Power transmission device"],
    ["Simulated jet-engines data", "Railway track geometry", "Cars production line", "Proton exchange membrane fuel cell", "Lithium-ion battery", "Circuit breakers", "On-load tap changers", "Oil condition", "Dehumidifier", "Rolling bearings", "Aircraft bearings", "LED Lighting", "Turbofan engine", "Semiconductor manufacturing station", "Tank reactor", "Naval machinery", "Mining machinery", "Laser device", "Steam Turbine", "Low methane compressors", "Cutter tool", "Simulated gas turbine", "Unknown Item", "Rectifier system", "Lead-acid batteries", "Lead bismuth eutectic experimental accelerator driven system", "Engine", "Vehicle suspension", "Wind turbines", "Electric cooling fan health prognostic", "2008 IEEE PHM challenge problem", "Carbon epoxy coupons", "Infrastructure", "Railway track circuit", "Filters", "Continuous stirred-tank reactor (CSTR)", "Production plant machines", "Aircraft engines", "Rotor shafts", "Pipelines", "Electrolytic capacitors", "Solder elements", "Switching power converters", "Polymer Electrolyte Membrane Fuel Cell", "Aerial Bundled Cables", "Rechargeable batteries with capacity regeneration phenomena", "Metallurgical Ladles", "High-voltage asymmetrical pulse (HVAP) track circuit", "Solenoid valves", "Sliding chair of railway point machine", "Sensors in interior permanent-magnet synchronous motor", "Gearbox in wind turbine", "Rectifier in railway electrical traction systems", "Actuator", "Supercapacitor"],  # Empty for manual text entry
    ["Online", "Offline"],
    ["Time series", "Structured text-based","Signals"]
]
# Create a list to hold the dropdowns
dropdowns = []
weight_entries = []
# Crear etiquetas, dropdowns y campos de entrada para los pesos
for i, label_text in enumerate(labels_input):
    label = ctk.CTkLabel(frame, text=label_text, font=("Arial", 14))
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e")  # Alinear a la derecha

    if dropdown_options[i]:
        dropdown = ctk.CTkComboBox(frame, values=dropdown_options[i])
        dropdown.grid(row=i, column=1, padx=10, pady=5)
        dropdowns.append(dropdown)  # Guardar el dropdown
    else:
        entry = ctk.CTkEntry(frame)
        entry.grid(row=i, column=1, padx=10, pady=5)
    
    # Campo de entrada para los pesos (con valor por defecto de 1.0)
    weight_entry = ctk.CTkEntry(frame, width=50)
    weight_entry.insert(0, "1.0")
    weight_entry.grid(row=i, column=2, padx=10, pady=5)
    weight_entries.append(weight_entry)  # Guardar el campo de entrada de peso

# Crear un label donde se mostrará la información
label_resultado = ctk.CTkLabel(app, text="")
label_resultado.pack(pady=30)
# Función para guardar los valores de los dropdowns y los pesos
shared_data = {}

def apply_algorithms():
    global shared_data
    
    DescriptionList,SolutionList=DescriptionsAndSolutions(CaseBase)
    Nested_Descriptions,Nested_Solutions=ModifiedCNN(DescriptionList,SolutionList,[0.05,0.325,0.325,0.05,0.25],[0.35,0.35,0.2,0.05,0.05],0.9,0.9)
    
    shared_data['Nested_Descriptions'] = Nested_Descriptions
    shared_data['Nested_Solutions'] = Nested_Solutions



def save_data():
    global shared_data

    values = [dropdown.get() for dropdown in dropdowns]  # Obtener los valores de los dropdowns
    weights = [weight.get() for weight in weight_entries]  # Obtener los valores de los pesos
    weights=[eval(i) for i in weights]

    Nested_Descriptions = shared_data.get('Nested_Descriptions', [])
    Nested_Solutions = shared_data.get('Nested_Solutions', [])
 
    Weights= [0.2, 0.2, 0.2, 0.2, 0.2]
    NumberRetrievals=5

    ListRetrievals=SearchSimilarModCNN(values,Nested_Descriptions,Nested_Solutions,NumberRetrievals,Weights)

    for i in range(len(ListRetrievals)):
        print(f"Solución {i} : ", ListRetrievals[i].solution,"\n")

    # Función para mostrar la información de los objetos en la GUI
    Div=Diversity(ListRetrievals,Weights)
    print("Diversidad de soluciones:", Div)
    print(len(Nested_Descriptions),len(Nested_Solutions))

    label_resultado.configure(text="")
    info_texto = ""
    for obj in ListRetrievals:
        info_texto += f"Solucion: {obj.solution}\n"
    
    # Actualiza el texto en el label
    label_resultado.configure(text=info_texto)
    # Aquí podrías guardar estos valores en un archivo o procesarlos como desees

# Crear un botón para guardar los datos
save_button = ctk.CTkButton(app, text="Submit Query", command=save_data)
save_button.pack(pady=10)

Apply_button = ctk.CTkButton(app, text="Apply Algorithm", command=apply_algorithms)
Apply_button.pack(pady=14)


# Ejecutar la aplicación
app.mainloop()
