import tkinter as tk

def chino_simple_cifrado(mensaje):
    mensaje = mensaje.upper().replace(" ", "") # Convertir a mayúsculas y eliminar espacios en blanco
    
    # Agregar una X al final si la longitud es impar
    if len(mensaje) % 2 != 0:
        mensaje += "X"
    
    # Dividir el mensaje en grupos de 2 letras
    grupos = [mensaje[i:i+2] for i in range(0, len(mensaje), 2)]
    
    # Reorganizar las letras de cada grupo y concatenarlos para obtener el texto cifrado
    texto_cifrado = ""
    for grupo in grupos:
        texto_cifrado += grupo[1] + grupo[0]
    
    return texto_cifrado

def chino_simple_descifrado(mensaje_cifrado):
    # Invertir la reorganización de letras en cada grupo y concatenarlos para obtener el mensaje original
    mensaje_descifrado = ""
    for i in range(0, len(mensaje_cifrado), 2):
        mensaje_descifrado += mensaje_cifrado[i+1] + mensaje_cifrado[i]
    
    return mensaje_descifrado

def cifrar():
    mensaje = entrada_mensaje.get()
    texto_cifrado = chino_simple_cifrado(mensaje)
    etiqueta_resultado.configure(text=texto_cifrado)

def descifrar():
    mensaje_cifrado = entrada_mensaje.get()
    mensaje_descifrado = chino_simple_descifrado(mensaje_cifrado)
    etiqueta_resultado.configure(text=mensaje_descifrado)

# Crear la ventana y los widgets
ventana = tk.Tk()
ventana.title("Cifrado Chino Simple")
etiqueta_mensaje = tk.Label(ventana, text="Mensaje:")
entrada_mensaje = tk.Entry(ventana)
etiqueta_resultado = tk.Label(ventana, text="")
etiqueta_resultado.config(font=("Courier", 12))

# Crear el menú
menu = tk.Menu(ventana)
ventana.config(menu=menu)

# Agregar opciones al menú
operaciones_menu = tk.Menu(menu)
menu.add_cascade(label="Operaciones", menu=operaciones_menu)
operaciones_menu.add_command(label="Encriptar", command=cifrar)
operaciones_menu.add_command(label="Desencriptar", command=descifrar)

# Colocar los widgets en la ventana
etiqueta_mensaje.grid(row=0, column=0, padx=5, pady=5)
entrada_mensaje.grid(row=0, column=1, padx=5, pady=5)
etiqueta_resultado.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Iniciar el bucle de eventos de la ventana
ventana.mainloop()
