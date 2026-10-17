# Tractor 3D — Los Podadores

Primera simulación articulada del tractor en **Gazebo Harmonic**, con dirección
delantera Ackermann, tracción trasera, conducción por teclado y recorridos
automáticos con obstáculos. Los escenarios son locales; no necesitan descargar
modelos de Fuel.

Esta etapa utiliza Gazebo Transport directamente. La conducción geométrica no
requiere ROS 2, YOLO ni PPO. El visor opcional de cámara usa YOLO en Windows.
Más adelante se puede integrar ROS 2 Jazzy y `ros_gz` sobre esta misma combinación
de Ubuntu 24.04 / Gazebo Harmonic.

## 1. Instalación en Windows: Ubuntu 24.04 dentro de WSL 2

El equipo inspeccionado ya tiene WSL 2 con una distribución llamada `Ubuntu`,
versión 26.04. No es la distribución objetivo de estos scripts: se conserva tal
cual. La simulación se prepara para una instalación **adicional** de Ubuntu 24.04,
soportada por los binarios oficiales de Gazebo Harmonic.

En **PowerShell**, instala la distribución adicional:

```powershell
wsl --install -d Ubuntu-24.04
```

Completa la creación del usuario y contraseña de Linux. Si Windows pide reiniciar,
hazlo antes de continuar. Comprueba que esta distribución usa WSL 2:

```powershell
wsl --list --verbose
```

Si `Ubuntu-24.04` aparece con versión 1:

```powershell
wsl --set-version Ubuntu-24.04 2
```

Abre **Ubuntu-24.04** (o `wsl -d Ubuntu-24.04`) y ejecuta:

```bash
cd /mnt/c/Users/luisg/OneDrive/Documentos/Universidad/PODADORES/automation
bash sim/tractor/setup_ubuntu.sh
```

El instalador pedirá la contraseña de Linux mediante `sudo`. Instala:
`gz-harmonic`, `python3-gz-transport13`, `python3-gz-msgs10`, `python3-tk` y
`python3-numpy`, desde
los repositorios de Ubuntu y el repositorio oficial de Gazebo.
No ejecuta el entrenamiento ni instala paquetes en el Python de Windows.

WSLg permite mostrar las ventanas de Gazebo y del teclado en Windows. Si no hay
pantalla gráfica, actualiza WSL desde PowerShell con `wsl --update` y reinicia la
sesión de Ubuntu. Revisa también el controlador gráfico de Windows. Esto no
requiere instalar un escritorio Linux completo.

## 2. Iniciar

Desde la misma carpeta de la repo en **Ubuntu-24.04**:

```bash
bash sim/tractor/run.sh --check
bash sim/tractor/run.sh
```

Se reconstruye el modelo desde el CAD y se validan los archivos con `gz sdf -k`.
Después se abren Gazebo y el panel de conducción. Puede tardar en aparecer la
escena la primera vez. En Gazebo, selecciona el modelo `tractor` en el árbol y
enfoca la cámara si no queda centrado.

Alternativamente, desde **PowerShell** en la repo:

```powershell
.\sim\tractor\start_windows.ps1 -Check
.\sim\tractor\start_windows.ps1
```

Si la política de ejecución impide abrir el `.ps1`, usa los comandos de Ubuntu;
no es necesario modificar permanentemente la política de Windows.

El lanzador establece `GZ_SIM_RESOURCE_PATH` y usa la partición de transporte
`los-podadores-tractor`, salvo que ya hayas definido `GZ_PARTITION`.
El simulador y las herramientas de diagnóstico deben usar la misma partición.

## 3. Conducir

Pon el foco en el **panel de teclado**, no en la ventana de Gazebo.

- **Enter:** habilitar el movimiento.
- **W / flecha arriba:** avanzar mientras se mantiene pulsado.
- **S / flecha abajo:** retroceder mientras se mantiene pulsado.
- **A / flecha izquierda:** dirigir las ruedas delanteras a la izquierda.
- **D / flecha derecha:** dirigir a la derecha.
- **W + A**, por ejemplo: avanzar describiendo una curva a la izquierda.
- **Soltar:** reducir progresivamente velocidad y centrar la dirección.
- **Espacio o botón DETENER:** enviar cero y deshabilitar. Enter habilita de nuevo.
- **Cambiar el foco:** enviar cero y deshabilitar.
- **Esc / cerrar:** cerrar el panel y la simulación iniciada por el lanzador.

Puedes orientar las ruedas estando detenido: el tractor no debe girar sobre su
centro. El panel muestra **órdenes**, no la velocidad medida del vehículo.

La parada es una orden a los controladores de velocidad de la simulación: no es
un modelo de frenos reales ni un sistema de seguridad para hardware. Al salir
normalmente se publica cero; si el proceso se mata forzosamente, Gazebo puede
conservar la última orden. Cierra la simulación o envía cero desde otra terminal.

## 4. Decisiones ante persona y perro

Para observar una **prueba corta autónoma en 3D**, desde Ubuntu-24.04 en la repo:

```bash
bash sim/tractor/run.sh --demo
```

Desde PowerShell también puedes usar:

```powershell
.\sim\tractor\start_windows.ps1 -Demo
```

Pulsa **Iniciar prueba** en el panel. No hace falta mantener W: el tractor sigue
una recta de prueba y una persona y un perro cruzan sucesivamente. El panel muestra:

- **CONTINUAR:** el espacio previsto para el tractor está libre.
- **FRENAR / ESPERAR:** hay un obstáculo en esa trayectoria; ordena detenerse.
- **ESPERAR**, al despejarse: comprueba durante 0,8 s que el paso siga libre.
- **SIN DATOS:** falta información actualizada; ordena detenerse.
- **FINALIZADO:** alcanzó el final del tramo de prueba, a 10,5 m.

El botón **DETENER** o Espacio interrumpe la marcha. Cierra el panel y vuelve a
lanzar el comando para repetir el escenario desde el inicio. Para conducir con
teclado y la misma asistencia de parada:

```bash
bash sim/tractor/run.sh --obstacles
```

Los cruces empiezan cuando el tractor alcanza posiciones definidas. La lógica
comprueba el volumen completo del tractor a lo largo de su trayectoria prevista,
incluidas curvas y marcha atrás. Usa un margen provisional de 0,65 m y velocidades
bajas. Por ahora la decisión ante una persona o un perro es **parar, esperar y
continuar**; todavía no planifica un desvío para rodearlos.

**La percepción usa posiciones y etiquetas conocidas de Gazebo. No es YOLO ni
reconocimiento de imágenes.** Las figuras son cuerpos geométricos móviles con
colisiones, sin animación de marcha ni apariencia realista; no constituyen un
dataset de entrenamiento visual. El adaptador observa únicamente los dos actores
del escenario. Este modo no detecta automáticamente cualquier objeto añadido.
Después se conectará una percepción basada en cámara y sensores, con estimación
de distancias, para alimentar las decisiones con observaciones medidas.

Esta recta corta comprueba los encuentros. Para los 110 m, la curva, la puerta y
el regreso, usa `--route`, descrito en la siguiente sección. Ese escenario usa
medidas provisionales para la puerta y el giro, pendientes de medir en campo.
Tampoco es una validación de frenado o seguridad del tractor físico.

El modo con obstáculos usa por defecto `GZ_PARTITION=los-podadores-tractor-obstacles`.
Para ejecutar la prueba sin ventanas:

```bash
bash sim/tractor/run.sh --headless --demo
```

Guarda el resultado en `runs/tractor_demo/last_run.json`. El comando devuelve éxito
solo si termina el tramo, registra paradas por ambos actores y mantiene el margen
geométrico configurado. El informe incluye los cambios de decisión y la separación
mínima observada; una ejecución exitosa no garantiza otros escenarios.

## 5. Recorrido completo: 110 m, curva, puerta y regreso

Con la instalación existente no necesitas paquetes adicionales. En Ubuntu-24.04,
desde la carpeta `automation`:

```bash
bash sim/tractor/run.sh --route
```

O desde PowerShell:

```powershell
.\sim\tractor\start_windows.ps1 -Route
```

Pulsa **Iniciar recorrido / Reanudar**. El tractor sigue la recta de 110 m, gira
90° a la izquierda, atraviesa una puerta abierta, da la vuelta en una zona amplia,
vuelve por la puerta y el camino y se detiene en el punto inicial, orientado hacia
el regreso. Toda la marcha es hacia delante: no gira sobre su centro ni retrocede
los 110 m. El panel muestra la fase, la decisión, el avance total, la velocidad
medida, los dos pasos por la puerta y un mapa con su posición. La cámara de Gazebo
se configura para seguir al tractor; Esc en la vista 3D permite soltar el seguimiento.
El botón **Seguir tractor con la cámara** lo vuelve a activar.

Una persona y un perro cruzan en la ida y otros dos en el regreso: son **cuatro
encuentros**. El tractor anticipa cruces usando la velocidad estimada de los actores,
se detiene, espera y reanuda al despejarse el paso. También comprueba los postes y
paneles laterales de la puerta. **DETENER / Espacio** pausa la marcha; el mismo
botón de inicio reanuda. Los actores ya activados pueden terminar de cruzar durante
la pausa. Cierra y vuelve a lanzar para reiniciar todo el escenario.

El entorno aproxima el camino con pavimento, marcas, cerco y árboles; las figuras
siguen siendo modelos geométricos sencillos. **No es una reconstrucción fotográfica
ni georreferenciada** del estadio. La puerta está abierta y no tiene mecanismo
automático de apertura.

Las medidas editables están en `sim/tractor/route.toml`:

- Recta: **110 m**, según la aproximación indicada.
- Curva: radio provisional **6 m**; permite respetar el límite de dirección del CAD.
- Tramo tras la curva: **5 m**, con la puerta a **3 m** de su inicio.
- Puerta: ancho libre provisional **3,5 m**.
- Zona posterior: **12 m** adicionales antes de iniciar el retorno.
- Retorno: tres arcos de radio **6 m** que cambian el sentido; reserva provisional
  de unos **18 × 23 m**, incluyendo el tractor y margen alrededor de los arcos.
- Velocidades provisionales: **1,2 m/s** en recta, **0,6 m/s** en curvas y
  **0,45 m/s** al aproximarse a la puerta.

El radio, la puerta, los anchos y el espacio de giro **todavía deben medirse en el
lugar real**. El recorrido completo ronda **315 m** contando ida, regreso, curva,
accesos y maniobra de retorno. No cabe suponer que una curva de 5 m de longitud
baste para girar este tractor: el programa comprueba el radio y el ancho libre
contra la geometría y el margen configurado.

La percepción y localización siguen usando la verdad conocida de Gazebo; no se ha
conectado YOLO ni entrenado una política de conducción. El control sigue una ruta
geométrica con realimentación de la pose física. Cerco y árboles tienen colisiones,
pero la percepción de este escenario observa los cuatro actores y la puerta;
no es un detector general de obstáculos añadidos. Si pierde datos, supera la
desviación lateral permitida o pierde el margen de separación, ordena parar.

La ejecución normal tarda varios minutos. Para adelantar el tiempo simulado sin
cambiar las velocidades físicas configuradas:

```bash
bash sim/tractor/run.sh --route --time-scale 4
```

Prueba automática sin ventanas:

```bash
bash sim/tractor/run.sh --route --headless --time-scale 4
```

La partición por defecto es `los-podadores-tractor-route`. Se guardan
`runs/tractor_route/last_run.json` y `trajectory.json`. Solo devuelve éxito si
completa la ruta, atraviesa la puerta dos veces, registra una parada medida ante
cada actor, mantiene el margen configurado y termina a menos de **0,2 m** del
inicio con la orientación de regreso. Esto valida este escenario simulado; no
valida los frenos, la detección visual o la seguridad del tractor físico.

## 6. Cámara RGB-D y tu YOLO entrenado

El tractor incorpora una cámara frontal RGB-D virtual a 1,25 m de altura:
**640 × 360**, **10 Hz**, alcance de profundidad de **0,15 a 25 m**. Se activa con
`--camera`; el modo normal no ejecuta el renderizado de sensores.

La integración usa el modelo existente `models/translate/best.pt`, que en este
proyecto contiene **Person (3)** y **Dog (10)**. El código resuelve los identificadores
por los nombres del modelo para admitir también pesos COCO con otra numeración.
No vuelve a entrenar, no descarga pesos y no modifica el modelo original.

En este equipo, PyTorch, Ultralytics y la GPU ya están preparados en el entorno de
**Windows**. Gazebo permanece en Ubuntu. Las imágenes se intercambian mediante
archivos locales de último fotograma, con sustitución atómica, sin abrir puertos
de red ni acumular una cola de imágenes.

### Abrir todo desde PowerShell

Cierra el panel y Gazebo anteriores. Si estás dentro de Ubuntu, `exit` vuelve a
PowerShell. Desde la carpeta `automation`:

```powershell
.\.venv\Scripts\python.exe sim/tractor/vision.py --launch
```

También existe el acceso directo:

```powershell
.\sim\tractor\start_vision.ps1
```

Se abren Gazebo, el panel del recorrido y el visor **CÁMARA + YOLO · SOLO
OBSERVACIÓN**. Pulsa **Iniciar recorrido / Reanudar** en el panel del tractor.
El visor muestra las imágenes que ve la cámara frontal y las cajas de detección,
clase y confianza de YOLO. Si hay una detección con profundidad válida, añade
la mediana de profundidad de la región central del recuadro (`prof. ROI`). Ese
valor puede incluir fondo y todavía no se valida como distancia de seguridad.
No se inventa una distancia cuando la profundidad falta o es inválida.

Cada ejecución crea una sesión separada en `runs/tractor_camera/<sesión>/`.
Cerrar el visor solicita cerrar solo la simulación iniciada por esa sesión.
El panel del tractor conserva su botón de parada y sigue controlando el recorrido.
Si la política de PowerShell bloquea el `.ps1`, usa el comando directo de Python;
no necesitas cambiar la política de Windows.

### Qué controla realmente al tractor

**YOLO observa; todavía no controla la marcha ni autoriza el paso.** La parada del
recorrido continúa usando los actores y la puerta conocidos de Gazebo. El visor
no se suscribe a esas etiquetas ni a sus posiciones: sus detecciones salen de
las imágenes. Las salidas incluyen `mode: monitor_only` y no se conectan a los
topics de dirección o tracción. Un fotograma sin detecciones no demuestra que
el camino esté libre; el visor marca la imagen como desactualizada si deja de llegar.

El modelo ya detectó personas en el video real `PODADORAS.mp4` y un perro en una
imagen del conjunto de prueba ROD. **Los actores actuales del simulador son
figuras geométricas** y no tienen la apariencia de esas imágenes reales. En una
captura probada, el perro geométrico no fue detectado. La conexión cámara → YOLO
funciona, pero todavía falta validar reconocimiento fiable dentro del simulador
con actores realistas o entrenamiento adicional con imágenes simuladas.

Antes de usar visión para frenar: validar personas/perros en distintos tamaños,
poses y luces; comprobar distancias; añadir seguimiento temporal y cubrir los
obstáculos que quedan fuera de la cámara frontal. La puerta y otros objetos no
se vuelven detectables automáticamente por filtrar solo Person/Dog. La cámara
RGB-D virtual es una elección para experimentar, no una compra de sensor ya decidida.

### Probar tu video o una cámara USB

Solo el visor YOLO, desde PowerShell:

```powershell
.\.venv\Scripts\python.exe sim/tractor/vision.py --source C:/Users/luisg/Videos/PODADORAS.mp4
.\.venv\Scripts\python.exe sim/tractor/vision.py --source 0
```

`0` es el índice de una cámara conectada a Windows. Un video o webcam RGB no
aportan profundidad; se muestran detecciones sin distancia. Puedes ajustar
`--confidence 0.45` o indicar otro archivo local con `--weights`.

### Ejecución separada y comprobación

Si prefieres dos terminales, en Ubuntu:

```bash
bash sim/tractor/run.sh --route --camera
```

En PowerShell, en `automation`:

```powershell
.\.venv\Scripts\python.exe sim/tractor/vision.py
```

En este modo el visor no cierra el Gazebo abierto aparte. Deben compartir la
carpeta `runs/tractor_camera/live`. El modo `--launch` gestiona esto automáticamente
con una carpeta y partición de transporte propias.

Se guardan `frame.npz` (último par RGB-D), `detections.json`, `last_frame.jpg`,
`summary.json`, `camera_status.json` al cerrar el puente y `gazebo.log` al usar
`--launch`. Las detecciones guardadas son históricas: consulta su marca de tiempo,
no las interpretes como una señal actual de vía libre. Los contadores de resumen
cuentan detecciones por fotograma, no individuos únicos. El intercambio queda
en `runs/`, excluido de Git.

Para una prueba sin visor sobre una sesión que esté emitiendo imágenes:

```powershell
.\.venv\Scripts\python.exe sim/tractor/vision.py --frames-dir runs/tractor_camera/live --headless --max-frames 5 --timeout 30
```

No hace falta instalar ROS 2 ni una segunda copia de PyTorch en Ubuntu. El puente
solo necesita NumPy y los bindings de Gazebo; `setup_ubuntu.sh` incluye
`python3-numpy` para instalaciones nuevas. El renderizado de sensores necesita un
backend gráfico disponible; en el equipo se probó con WSLg.

Para comprobar toda la cadena y el cierre de la sesión desde un solo comando,
sin ventanas:

```powershell
.\.venv\Scripts\python.exe sim/tractor/vision.py --launch --headless --max-frames 5 --timeout 40
```

## 7. Qué se tomó del CAD y qué falta calibrar

`cad/TRACTOR.stp` es una copia del archivo original proporcionado. Se conservan
su hash SHA-256 y las medidas extraídas en `cad_dimensions.json`.

- Distancia entre ejes: **2,00 m**.
- Ruedas traseras: radio **0,50 m**, ancho **0,27 m**, separación entre centros **1,40 m**.
- Ruedas delanteras: radio **0,37 m**, ancho **0,16 m**, separación entre centros **1,29 m**.
- Ejes y sección del larguero obtenidos de los círculos y vértices del STEP.
- Conversión de milímetros a metros; X hacia delante, Y a la izquierda, Z arriba.

El STEP contiene un sólido unido, sin articulaciones de robot. La versión móvil
es una **reconstrucción analítica** con cilindros separados para ruedas y ejes;
el larguero central se aproxima con una caja. No es una conversión exacta de
todo el BREP a mallas ni incluye la carrocería, asiento o detalles de la foto real.
La diferencia de altura original entre las ruedas deja aproximadamente 1,42 mm
de separación inicial bajo las delanteras; se conserva y el vehículo se asienta
con la gravedad.

`parameters.toml` contiene valores **provisionales** de masa, fricción, velocidad,
aceleración y dirección. No representan componentes ya elegidos. Las inercias se
aproximan con cuerpos simples y masa concentrada para el chasis. El modelo es
rígido, sin suspensión ni simulación de motor eléctrico, batería o neumático
deformable. La calibración del tractor real será una etapa posterior.

La dirección usa `AckermannSteering` en modo `steering_only`. La tracción usa
`DiffDrive` únicamente para distribuir las velocidades del diferencial trasero,
con `yaw_rate = velocidad * tan(dirección) / distancia_entre_ejes`. Las ruedas
delanteras ruedan libremente. Esto no convierte al vehículo en uno que gire
sobre sí mismo ni supone que el tractor real tendrá dos motores de tracción.

## 8. Regenerar y comprobar

Para comprobar geometría, articulaciones, inercias y órdenes de movimiento sin
tener Gazebo instalado:

```bash
python3 sim/tractor/build_model.py
python3 -m unittest discover -s sim/tractor/tests -v
```

En Windows, se puede usar el Python del proyecto:

```powershell
uv run python sim/tractor/build_model.py
uv run python -m unittest discover -s sim/tractor/tests -v
```

Si `gz` está disponible, la suite también valida el modelo y el escenario con
`gz sdf -k`; esas cuatro pruebas se omiten cuando Gazebo no está instalado.
**Estas comprobaciones no prueban por sí solas que el motor físico, las
colisiones y la interfaz gráfica funcionen en el equipo.** Después
de instalar Gazebo, realiza la validación física con:

```bash
bash sim/tractor/run.sh
```

Comprueba recta, curva a ambos lados, marcha atrás y parada. Las ruedas delanteras
deben alcanzar ángulos distintos en curva. En otra terminal Ubuntu:

```bash
export GZ_PARTITION=los-podadores-tractor
gz topic -e -t /model/tractor/joint_state
gz topic -e -t /model/tractor/wheel_odometry
```

La odometría procede del diferencial trasero: es una estimación de ruedas,
no una medición de GPS ni la verdad física cuando hay deslizamiento.
Para enviar una parada manual desde esa terminal:

```bash
gz topic -t /model/tractor/rear_axle/cmd_vel -m gz.msgs.Twist -p 'linear: {x: 0}, angular: {z: 0}'
gz topic -t /model/tractor/steering_angle -m gz.msgs.Double -p 'data: 0'
```

Para arrancar únicamente el servidor y publicar órdenes desde otra herramienta:

```bash
bash sim/tractor/run.sh --headless
```

## Referencias oficiales

- [Instalar Gazebo Harmonic en Ubuntu](https://gazebosim.org/docs/harmonic/install_ubuntu/)
- [Combinaciones Gazebo / ROS 2](https://gazebosim.org/docs/harmonic/ros_installation/)
- [Controlador Ackermann de Gazebo](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1AckermannSteering.html)
- [Controlador del diferencial trasero](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1DiffDrive.html)
- [Gazebo Transport en Python](https://gazebosim.org/api/transport/13/python.html)
- [Sensores RGB-D de Gazebo](https://gazebosim.org/api/sensors/7/classgz_1_1sensors_1_1RgbdCameraSensor.html)
- [Inferencia de Ultralytics YOLO](https://docs.ultralytics.com/modes/predict/)
- [Configuración de seguimiento de cámara de Gazebo](https://github.com/gazebosim/gz-gui/blob/gz-gui8/src/plugins/camera_tracking/CameraTracking.cc)
- [Aplicaciones gráficas en WSL](https://learn.microsoft.com/es-es/windows/wsl/tutorials/gui-apps)
