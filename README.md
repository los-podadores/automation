# Los Podadores

## Instalación y Configuración

Se recomienda utilizar [uv](https://github.com/astral-sh/uv), un instalador y gestor de paquetes de Python.

### 1. Instalar `uv`

#### macOS y Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### A través de `pip`
```bash
pip install uv
```

### 2. Configuración del Entorno de Python

Una vez instalado `uv`, puedes inicializar el entorno virtual e instalar las dependencias del proyecto ejecutando:

```bash
uv sync
```

Este comando leerá las especificaciones de [pyproject.toml](file:///home/cricro/projects/los-podadores/automation/pyproject.toml) y creará un entorno local en la carpeta `.venv/`.

#### Integración con Nix
Si utilizas el gestor de paquetes Nix, se proporciona un entorno de desarrollo en [flake.nix](file:///home/cricro/projects/los-podadores/automation/flake.nix). Este configura automáticamente las rutas de librerías (`LD_LIBRARY_PATH` para OpenGL, SDL2, glib, wayland, etc.) y `uv`:

```bash
# Entrar al shell de desarrollo
nix develop

# O si utilizas direnv
direnv allow
```

---

## El Entorno de Aprendizaje por Refuerzo (`src/rl/`)

El objetivo del robot es cubrir la mayor cantidad posible de área transitable en un campo generado aleatoriamente, evitando colisionar con los límites del mapa y obstáculos internos.

### Espacio de Acciones
* Velocidad lineal objetivo: Los valores positivos avanzan hacia adelante, los valores negativos retroceden a la mitad de velocidad.
* Dirección angular objetivo: Los valores positivos giran a la izquierda, los valores negativos giran a la derecha.

### Espacio de Observaciones
Las observaciones se devuelven en un diccionario, estructurado para proporcionar "mapas espaciales egocéntricos multiescala" y lecturas de sensores directos

### Función de Recompensa
La señal de recompensa considera los siguientes factores:
* Recompensa por Cobertura de Área: Recompensa positiva basada en el número de nuevos píxeles cubiertos.
* Penalización por Variación Total (TV): Penaliza cambios bruscos en el mapa de cobertura local, evitando que el robot deje pequeños huecos sin cortar.
* Penalización por Colisión: Penalización negativa grande si el robot choca contra límites u obstáculos.
* Progreso de Frontera: Si no se cubre nueva área, se recompensa o penaliza al agente en función de si se aproxima o se aleja a la siguiente celda vacía.
* Penalizaciones por Paso de Tiempo: Una penalización fija por cada paso de tiempo, forzando al agente a resolver el entorno con rapidez.

### Arquitectura de Red
Dado que las posiciones espaciales en diferentes escalas de los mapas no coinciden píxel a píxel, las convoluciones estándar mezclarían información con escalas incorrectas.

Por ello, se utiliza una arquitectura CNN agrupada por escalas (SGCNN) en el extractor de características ([StackedMapFeaturesExtractor](file:///home/cricro/projects/los-podadores/automation/src/rl/architectures.py#L11)):

### Aprendizaje con Currículum
El entorno incrementa la complejidad del mapa de forma incremental en 8 fases para facilitar el aprendizaje:
* Fase 1: Campos pequeños (radios de 2.5–6.0m), 1-2 obstáculos y un límite máximo de 3500 pasos.
* Fase 8: Campos grandes (radios de 16.0–18.0m), 7-9 obstáculos y un límite máximo de 10000 pasos.

El script de entrenamiento avanza automáticamente de fase cuando la tasa de éxito de evaluación acumulada supera el `80%` (calculado sobre una ventana de 50 episodios).

## Comandos y Uso

Todos los comandos se pueden ejecutar anteponiendo `uv run <script>` para asegurar que se ejecuten dentro del entorno virtual del proyecto.

### 1. Teleoperación y Depuración Manual
Ejecuta la interfaz interactiva basada en Pygame para conducir manualmente el robot y visualizar en tiempo real las lecturas de los sensores, los mapas y la cortadora.

```bash
uv run python src/rl/teleop_debug.py
```

### 2. Entrenar el Agente de RL
Entrena una política PPO utilizando el currículum del entorno:

```bash
# Iniciar entrenamiento desde cero
uv run python src/rl/train.py
```

Los logs del entrenamiento se guardan en `logs/v3/` (compatibles con TensorBoard) y los checkpoints se almacenan en `models/v3/`.

### 3. Visualizar el Modelo Entrenado Final
El modelo completamente entrenado de forma óptima está copiado en la raíz del proyecto como [final_model.zip](file:///home/cricro/projects/los-podadores/automation/final_model.zip). Se puede probar con:

```bash
uv run python src/rl/visualize.py --model final_model.zip --phase 3 --episodes 5
```
