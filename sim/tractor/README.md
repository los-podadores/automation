# Tractor 3D — Los Podadores

Primera simulación articulada del tractor en **Gazebo Harmonic**, con dirección
delantera Ackermann, tracción trasera y un panel de teclado. El escenario incluye
suelo plano, límites y obstáculos locales; no necesita descargar modelos de Fuel.

Esta etapa utiliza Gazebo Transport directamente. **ROS 2, YOLO y PPO no son
necesarios todavía.** Más adelante se puede integrar ROS 2 Jazzy y `ros_gz` sobre
esta misma combinación de Ubuntu 24.04 / Gazebo Harmonic.

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
`gz-harmonic`, `python3-gz-transport13`, `python3-gz-msgs10` y `python3-tk`, desde
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

## 4. Qué se tomó del CAD y qué falta calibrar

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

## 5. Regenerar y comprobar

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
`gz sdf -k`; esas dos pruebas se omiten cuando Gazebo no está instalado.
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
- [Aplicaciones gráficas en WSL](https://learn.microsoft.com/es-es/windows/wsl/tutorials/gui-apps)
