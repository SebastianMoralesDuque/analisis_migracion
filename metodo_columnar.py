import tkinter as tk

def cifrado_columnar_simple(mensaje, clave):
    mensaje = mensaje.upper().replace(" ", "") # Convertir a mayúsculas y eliminar espacios en blanco
    clave = clave.upper().replace(" ", "") # Convertir a mayúsculas y eliminar espacios en blanco
    
    # Calcular el número de filas necesarias para acomodar todas las letras del mensaje
    num_filas = len(mensaje) // len(clave)
    if len(mensaje) % len(clave) != 0:
        num_filas += 1
    
    # Añadir caracteres de relleno (X) si es necesario para completar la última columna
    num_caracteres_restantes = num_filas * len(clave) - len(mensaje)
    mensaje += "X" * num_caracteres_restantes
    
    # Reorganizar las letras del mensaje en una matriz según la clave
    matriz = []
    for i in range(num_filas):
        fila = []
        for j in range(len(clave)):
            index = i * len(clave) + j
            fila.append(mensaje[index])
        matriz.append(fila)
        
    # Ordenar las columnas de la matriz según la clave
    clave_ordenada = sorted(clave)
    permutacion = [clave.index(letra) for letra in clave_ordenada]
    matriz_ordenada = []
    for i in permutacion:
        columna = [fila[i] for fila in matriz]
        matriz_ordenada.append(columna)
    
    # Concatenar las columnas de la matriz ordenada para obtener el texto cifrado
    texto_cifrado = "".join([letra for columna in matriz_ordenada for letra in columna])
    
    return texto_cifrado

def cifrar():
    mensaje = entrada_mensaje.get()
    clave = entrada_clave.get()
    texto_cifrado = cifrado_columnar_simple(mensaje, clave)
    etiqueta_resultado.configure(text=texto_cifrado)

# Crear la ventana y los widgets
ventana = tk.Tk()
ventana.title("Cifrado Columnar Simple")
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

# Iniciar la aplicación
ventana.mainloop()
