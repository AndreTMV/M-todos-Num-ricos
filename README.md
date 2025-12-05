# Visualizador de Métodos Numéricos

Este proyecto es una herramienta interactiva escrita en Python para resolver ecuaciones no lineales utilizando métodos numéricos clásicos. Además de calcular la raíz, genera **animaciones explicativas** que muestran paso a paso cómo funciona cada método gráficamente.

---

## Instrucciones para correr el proyecto 

Si nunca has usado Python, sigue estos pasos sencillos para echar a andar el proyecto en tu computadora.

### 1. Instalar Python
Primero necesitas tener Python instalado.
- Ve a [python.org/downloads](https://www.python.org/downloads/) y descarga la última versión para tu sistema operativo.
- Al instalar, asegúrate de marcar la casilla que dice **"Add Python to PATH"** (esto es muy importante).
- Si usas Mac, puedes usar Homebrew para instalar Python. (brew install python)

### 2. Descargar este repositorio
Clona el repositorio en tu computadora.

### 3. Instalar las dependencias (librerías necesarias)
Este proyecto usa librerías externas para hacer los cálculos y las gráficas. Necesitamos instalarlas dentro de un ambiente virtual.

1.  Abre una terminal (en Windows busca "CMD" o "PowerShell", en Mac/Linux abre "Terminal").
2.  Navega hasta la carpeta donde guardaste el proyecto. Puedes usar el comando `cd` seguido de la ruta de la carpeta.
    *   Ejemplo: `cd Documentos/mi-proyecto-metodos`
3.  Una vez dentro de la carpeta, escribe el siguiente comando y presiona Enter:

    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    *Esto creara el ambiente virtual y descargará e instalará automáticamente todo lo necesario.*

### 4. Ejecutar el programa
Para iniciar el programa, escribe en la terminal:

```bash
python main.py
```

Sigue las instrucciones que aparecen en pantalla para ingresar tu ecuación y elegir el método.

---

## 📂 Descripción de los archivos

Aquí tienes un resumen de qué hace cada archivo en este proyecto:

### Programa Principal
*   **`main.py`**: Es el cerebro del proyecto. Es el archivo que ejecutas para iniciar el programa. Se encarga de preguntarte la función, el método que quieres usar y los parámetros iniciales. Luego llama a los algoritmos correspondientes.
*   **`main_d.py`**: Una versión alternativa de `main.py` que contiene mucha más documentación y comentarios explicativos en el código. Útil si quieres leer el código para entender cómo está construido el flujo principal.
*   **`requirements.txt`**: Una lista de ingredientes. Le dice a Python qué librerías extra necesita descargar para que todo funcione.

### Algoritmos (Métodos Numéricos)
Cada uno de estos archivos contiene la lógica matemática de un método específico y su función para crear la animación:

*   **`biseccion.py`**: Implementa el **Método de Bisección**, que divide intervalos a la mitad repetidamente para encontrar la raíz.
*   **`posicion_falsa.py`**: Implementa el **Método de la Posición Falsa (Regula Falsi)**, similar a bisección pero usando líneas rectas para estimar mejor el siguiente punto.
*   **`secante.py`**: Implementa el **Método de la Secante**, que usa líneas secantes entre puntos para aproximarse a la raíz.
*   **`newton.py`**: Implementa el **Método de Newton-Raphson**, uno de los más rápidos, que usa derivadas (tangentes) para encontrar la raíz.
*   **`punto_fijo.py`**: Implementa el **Método de Punto Fijo**, que transforma la ecuación a la forma $x = g(x)$ para iterar hacia la solución.

### Herramientas y Utilerías
*   **`utils.py`**: Contiene funciones de ayuda que usan todos los demás archivos. Aquí está la magia para leer las fórmulas matemáticas que escribes, verificar si el método ya terminó (criterios de paro) y generar las imágenes para las animaciones.
*   **`utils_d.py`**: Versión documentada o de desarrollo de las utilerías.

*(Nota: Los archivos que terminan en `_d.py` son variantes con más comentarios o detalles de implementación, pero la lógica principal está en los archivos sin `_d`).*

---

## 🧠 ¿Cómo funciona esto "a groso modo"?

El proyecto sigue un flujo lineal muy sencillo:

1.  **Entrada de Datos**: El programa (`main.py`) te pide que escribas una ecuación, por ejemplo `x**3 - 2*x - 5`. Usa una función especial en `utils.py` para "entender" ese texto y convertirlo en una función matemática real que Python pueda calcular.
2.  **Procesamiento**: Dependiendo del método que elijas (p.ej. Bisección), el programa llama a la función en el archivo correspondiente (p.ej. `biseccion.py`). Esta función ejecuta el algoritmo matemático en un bucle, repitiendo los cálculos hasta encontrar la raíz o alcanzar el límite de intentos.
3.  **Historial**: Mientras calcula, el programa va guardando una "foto" de cada paso (cuánto valía x, cuánto valía la función, el error, etc.) en una lista historial.
4.  **Visualización**: Si eliges ver la animación, el programa toma ese historial y usa la librería `matplotlib` para dibujar una gráfica por cada paso.
    *   Dibuja la curva de tu función.
    *   Dibuja puntos y líneas que muestran lo que hizo el método en ese paso específico.
    *   Genera un panel con las fórmulas matemáticas explicadas.
    *   Junta todo en una animación fluida (GIF).
