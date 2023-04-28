import tkinter as tk

def interrumpido_cifrado(mensaje, clave):
    mensaje = mensaje.upper().replace(" ", "") # Convertir a mayúsculas y eliminar espacios en blanco
    clave = clave.upper().replace(" ", "") # Convertir a mayúsculas y eliminar espacios en blanco

    # Calcular el número de filas necesarias para acomodar todas las letras del mensaje
    num_filas = len(mensaje) // len(clave)
    if len(mensaje) % len(clave) != 0:
        num_filas += 1

    # Añadir caracteres de relleno (X) si es necesario para completar la última fila
    num_caracteres_restantes = num_filas * len(clave) - len(mensaje)
    mensaje += "X" * num_caracteres_restantes

    # Reorganizar las letras del mensaje en filas según la clave
    filas = []
    for i in range(num_filas):
        fila = []
        for j in range(len(clave)):
            index = i * len(clave) + j
            fila.append(mensaje[index])
        filas.append(fila)
    
    # Obtener la permutación de columnas correspondiente a la clave
    clave_ordenada = sorted(clave)
    permutacion = [clave.index(letra) for letra in clave_ordenada]

    # Recorrer las columnas en el orden especificado por la permutación y concatenarlas para obtener el texto cifrado
    texto_cifrado = ""
    for i in permutacion:
        for j in range(num_filas):
            texto_cifrado += filas[j][i]
    
    return texto_cifrado

def cifrar():
    mensaje = entrada_mensaje.get()
    clave = entrada_clave.get()
    texto_cifrado = interrumpido_cifrado(mensaje, clave)
    etiqueta_resultado.configure(text=texto_cifrado)

# Crear la ventana y los widgets
ventana = tk.Tk()
ventana.title("Cifrado Interrumpido")
etiqueta_mensaje = tk.Label(ventana, text="Mensaje:")
entrada_mensaje = tk.Entry(ventana)
etiqueta_clave = tk.Label(ventana, text="Clave:")
entrada_clave = tk.Entry(ventana)
boton_cifrar = tk.Button(ventana, text="Cifrar", command=cifrar)
etiqueta_resultado = tk.Label(ventana, text="")
etiqueta_resultado.config(font=("Courier", 12))

# Colocar los widgets en la ventana
etiqueta_mensaje.grid(row=0, column=0, padx=5, pady=5)
entrada_mensaje.grid(row=0, column=1, padx=5, pady=5)
etiqueta_clave.grid(row=1, column=0, padx=5, pady=5)
entrada_clave.grid(row=1, column=1, padx=5, pady=5)
boton_cifrar.grid(row=2, column=0, padx=5, pady=5)
etiqueta_resultado.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Iniciar el bucle de la aplicación
ventana.mainloop()
