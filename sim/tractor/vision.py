"""YOLO perception for Gazebo RGB-D, videos or a Windows webcam.

Uses the existing project model and Windows GPU, without reading actor labels.
Only live Gazebo sessions can supply results to the optional braking interlock.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from camera_io import detection_depth, read_frame
from session_control import OperatorClient, read_json, write_json

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))


class VisionMonitor:
    def __init__(self, args):
        import torch
        from src.translate.detect import ObstacleDetector

        if not args.weights.is_file():
            raise FileNotFoundError(f"No existe el modelo: {args.weights}")
        self.args = args
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.detector = ObstacleDetector(
            args.weights, confidence=args.confidence, device=self.device
        )
        self.classes = [
            int(key)
            for key, name in self.detector.model.names.items()
            if name.casefold() in ("person", "dog")
        ]
        names = {self.detector.model.names[key].casefold() for key in self.classes}
        if names != {"person", "dog"}:
            raise ValueError("El modelo debe incluir clases Person/person y Dog/dog")
        self.output = (
            args.frames_dir
            if args.source is None
            else PROJECT / "runs/tractor_vision/video"
        )
        if args.output_dir is not None:
            self.output = args.output_dir
        self.output.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.updated = 0.0
        self.error = ""
        self.processed = 0
        self.counts = {"person": 0, "dog": 0}
        self.finished = False
        self.capture = None
        self.still_frame = None

    def run(self):
        source = self.args.source
        sequence = -1
        deadline = (
            time.monotonic() + self.args.timeout if self.args.timeout else float("inf")
        )
        last_saved = 0.0
        try:
            session_id = None
            if getattr(self.args, "vision_brake", False):
                session = read_json(self.args.frames_dir / "session.json")
                if (
                    source is not None
                    or not session
                    or not session.get("session_id")
                    or session.get("vision_brake") is not True
                ):
                    raise RuntimeError(
                        "La parada YOLO requiere la cámara de una sesión Gazebo"
                    )
                session_id = session["session_id"]
            is_photo = source is not None and Path(source).suffix.lower() in (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
                ".tif",
                ".tiff",
            )
            if is_photo:
                self.still_frame = cv2.imread(source)
                if self.still_frame is None:
                    raise RuntimeError(f"No se pudo leer la fotografía: {source}")
            elif source is not None:
                self.capture = cv2.VideoCapture(
                    int(source) if source.isdecimal() else source
                )
                if not self.capture.isOpened():
                    raise RuntimeError(f"No se pudo abrir el video/cámara: {source}")
                if self.args.start_seconds and not self.capture.set(
                    cv2.CAP_PROP_POS_MSEC, self.args.start_seconds * 1000
                ):
                    raise RuntimeError(
                        "No se pudo avanzar al segundo indicado del video"
                    )
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                depth = None
                if self.still_frame is not None:
                    bgr, stamp, sequence = self.still_frame, 0.0, 0
                elif self.capture is not None:
                    success, bgr = self.capture.read()
                    if not success:
                        break
                    sequence += 1
                    if sequence % self.args.stride:
                        continue
                    stamp = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                else:
                    try:
                        if (
                            time.time()
                            - (self.args.frames_dir / "frame.npz").stat().st_mtime
                            > 2
                        ):
                            self.stop_event.wait(0.05)
                            continue
                        rgb, depth, stamp, new_sequence = read_frame(
                            self.args.frames_dir
                        )
                    except (FileNotFoundError, PermissionError, ValueError, OSError):
                        self.stop_event.wait(0.05)
                        continue
                    if new_sequence == sequence:
                        self.stop_event.wait(0.025)
                        continue
                    sequence = new_sequence
                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                started = time.perf_counter()
                detections = self.detector.detect(bgr, classes=self.classes)
                annotated = self.detector.annotate(bgr, detections)
                boxes = []
                for detection in detections:
                    depth_m = (
                        detection_depth(depth, detection.bbox)
                        if depth is not None
                        else None
                    )
                    label = detection.class_name.casefold()
                    self.counts[label] += 1
                    boxes.append(
                        dict(
                            label=label,
                            confidence=detection.confidence,
                            bbox=list(detection.bbox),
                            roi_depth_m=depth_m,
                        )
                    )
                    if depth_m is not None:
                        x, y = detection.bbox[:2]
                        cv2.putText(
                            annotated,
                            f"prof. ROI {depth_m:.1f} m",
                            (x, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 230, 255),
                            1,
                        )
                self.processed += 1
                result = dict(
                    mode="vision_brake" if session_id else "monitor_only",
                    session_id=session_id,
                    control_source="YOLO front-image stop plus Gazebo geometric backup"
                    if session_id
                    else "Gazebo geometry; YOLO does not drive"
                    if source is None
                    else "None; real media inference only",
                    source="Gazebo rendered RGB-D" if source is None else source,
                    frame_sequence=sequence,
                    frame_time_s=stamp,
                    written_at=time.time(),
                    device=self.device,
                    weights=str(self.args.weights),
                    inference_ms=(time.perf_counter() - started) * 1000,
                    detections=boxes,
                )
                write_json(self.output / "detections.json", result)
                if time.monotonic() - last_saved >= 1 or self.processed == 1:
                    cv2.imwrite(str(self.output / "last_frame.jpg"), annotated)
                    last_saved = time.monotonic()
                with self.lock:
                    self.latest = annotated, result
                    self.updated = time.monotonic()
                if self.args.max_frames and self.processed >= self.args.max_frames:
                    break
                if self.still_frame is not None:
                    break
                self.stop_event.wait(0.05)
        except Exception as error:
            self.error = str(error)
        finally:
            if self.capture is not None:
                self.capture.release()
            self.finished = True
            (self.output / "summary.json").write_text(
                json.dumps(
                    dict(
                        mode="vision_brake"
                        if getattr(self.args, "vision_brake", False)
                        else "monitor_only",
                        processed_frames=self.processed,
                        detections_by_class=self.counts,
                        error=self.error,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


def launch_gazebo(args):
    if sys.platform != "win32":
        raise RuntimeError("--launch se ejecuta con el Python de Windows")
    session = uuid.uuid4().hex[:12]
    args.frames_dir = PROJECT / "runs/tractor_camera" / session
    args.frames_dir.mkdir(parents=True)
    write_json(
        args.frames_dir / "session.json",
        dict(session_id=session, vision_brake=getattr(args, "vision_brake", False)),
    )
    linux = subprocess.run(
        [
            "wsl.exe",
            "-d",
            args.distribution,
            "--",
            "wslpath",
            "-u",
            args.frames_dir.as_posix(),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    run_script = subprocess.run(
        [
            "wsl.exe",
            "-d",
            args.distribution,
            "--",
            "wslpath",
            "-u",
            (ROOT / "run.sh").as_posix(),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    log = (args.frames_dir / "gazebo.log").open("wb")
    try:
        command = [
            "wsl.exe",
            "-d",
            args.distribution,
            "--",
            "env",
            f"GZ_PARTITION=tractor-vision-{session}",
            "bash",
            run_script,
            "--route",
            "--camera",
            "--frames-dir",
            linux,
        ]
        if args.headless:
            command.append("--headless")
        if getattr(args, "vision_brake", False):
            command.append("--vision-brake")
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    finally:
        log.close()
    print(f"Sesión de cámara: {args.frames_dir}", flush=True)
    return process


def show_window(monitor, simulation):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Los Podadores · Cámara + YOLO")
    panel = ttk.Frame(root, padding=14)
    panel.grid()
    ttk.Label(
        panel,
        text="CÁMARA + YOLO · RECORRIDO"
        if monitor.args.source is None
        else "CÁMARA + YOLO · SOLO OBSERVACIÓN",
        font=("Sans", 14, "bold"),
    ).grid(pady=6)
    ttk.Label(
        panel,
        text="PARADA ACTIVA: persona o perro en cámara → frenar; 1,2 s sin detecciones → reanudar."
        if getattr(monitor.args, "vision_brake", False)
        else "La conducción y la parada siguen usando la geometría de Gazebo."
        if monitor.args.source is None
        else f"Fuente real: {Path(monitor.args.source).name if not monitor.args.source.isdecimal() else 'cámara ' + monitor.args.source}",
        wraplength=750,
    ).grid()
    ttk.Label(
        panel,
        text="Respaldo geométrico activo: YOLO aún reconoce mal las figuras simplificadas de Gazebo."
        if monitor.args.source is None
        else "No detectar algo no significa que el camino esté libre.",
    ).grid(pady=4)
    operator = (
        OperatorClient(monitor.args.frames_dir) if monitor.args.source is None else None
    )
    route_status = tk.StringVar(value="Conectando los controles del recorrido…")
    start_button = None
    if operator is not None:
        ttk.Label(
            panel, textvariable=route_status, font=("Sans", 11, "bold"), wraplength=620
        ).grid(pady=6)
        actions = ttk.Frame(panel)
        actions.grid(sticky="ew", pady=4)
        actions.columnconfigure((0, 1), weight=1)

        def send_control(action):
            try:
                operator.send(action)
            except OSError as error:
                route_status.set(f"No se pudo enviar la orden: {error}")

        start_button = ttk.Button(
            actions,
            text="INICIAR RECORRIDO / REANUDAR",
            command=lambda: send_control("start"),
            state="disabled",
        )
        start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(
            actions, text="DETENER — ESPACIO", command=lambda: send_control("stop")
        ).grid(row=0, column=1, sticky="ew")
    status = tk.StringVar(value="Esperando imágenes de la cámara…")
    ttk.Label(panel, textvariable=status, font=("Sans", 11, "bold")).grid(pady=6)
    picture = ttk.Label(panel)
    picture.grid()
    details = tk.StringVar(
        value="Person / Dog · Profundidad de ROI solo en Gazebo RGB-D"
        if monitor.args.source is None
        else "Person / Dog · Video o foto real, sin medición de distancia"
    )
    ttk.Label(panel, textvariable=details).grid(pady=6)
    ttk.Button(
        panel,
        text="Cerrar visor y simulación de esta sesión"
        if simulation is not None
        else "Cerrar visor",
        command=root.quit,
    ).grid(sticky="ew")
    worker = threading.Thread(target=monitor.run, daemon=True)
    worker.start()
    last_sequence = None
    last_heartbeat = 0.0

    def tick():
        nonlocal last_sequence, last_heartbeat
        if operator is not None:
            try:
                if time.monotonic() - last_heartbeat > 0.3:
                    operator.heartbeat()
                    last_heartbeat = time.monotonic()
                route = operator.status()
                if route is None:
                    route_status.set(
                        "Esperando conexión con el controlador del recorrido…"
                    )
                    start_button.configure(state="disabled")
                else:
                    speed = route.get("speed_mps")
                    route_text = (
                        f"{route['state']} · {route['reason']}\n{route['progress_m']:.1f} / {route['length_m']:.1f} m · Velocidad: {speed:.2f} m/s"
                        if speed is not None
                        else f"{route['state']} · {route['reason']}"
                    )
                    crossing = route.get("next_crossing")
                    if crossing is not None:
                        route_text += f"\nPróximo cruce programado: {crossing['label']} en {crossing['distance_m']:.0f} m"
                    route_status.set(route_text)
                    start_button.configure(
                        state="disabled"
                        if route["finished"] or route["failure"]
                        else "normal"
                    )
            except OSError as error:
                route_status.set(f"Sin conexión con los controles: {error}")
                start_button.configure(state="disabled")
        if simulation is not None and simulation.poll() is not None:
            status.set(
                "Gazebo terminó. Cierra este visor; revisa gazebo.log si hubo un error."
            )
        elif monitor.error:
            status.set(f"ERROR: {monitor.error}")
        elif monitor.finished and monitor.args.source is not None:
            status.set("Video / imagen terminado · Puedes cerrar el visor")
        elif time.monotonic() - monitor.updated > 1.5:
            status.set("SIN IMAGEN ACTUALIZADA · detecciones anteriores no son válidas")
        with monitor.lock:
            latest = monitor.latest
        if latest is not None and latest[1]["frame_sequence"] != last_sequence:
            bgr, result = latest
            last_sequence = result["frame_sequence"]
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            image.thumbnail((800, 450))
            picture.image = ImageTk.PhotoImage(image)
            picture.configure(image=picture.image)
            if time.monotonic() - monitor.updated <= 1.5:
                status.set(
                    f"YOLO activo · {len(result['detections'])} personas/perros detectados · {result['inference_ms']:.0f} ms · {'GPU' if monitor.device == '0' else 'CPU'}"
                )
            details.set(
                " | ".join(
                    f"{d['label']} {d['confidence']:.0%}" for d in result["detections"]
                )
                or "Sin detecciones Person/Dog en este fotograma"
            )
        root.after(60, tick)

    root.protocol("WM_DELETE_WINDOW", root.quit)
    if operator is not None:

        def emergency_stop(_event):
            send_control("stop")
            return "break"

        root.bind_class("VisionEmergencyStop", "<KeyPress-space>", emergency_stop)
        root.bind_class(
            "VisionEmergencyStop", "<KeyRelease-space>", lambda _event: "break"
        )

        def install_stop_binding(widget):
            widget.bindtags(("VisionEmergencyStop", *widget.bindtags()))
            for child in widget.winfo_children():
                install_stop_binding(child)

        install_stop_binding(root)
    root.after(100, tick)
    try:
        root.mainloop()
    finally:
        if operator is not None:
            try:
                operator.send("stop")
            except OSError:
                pass  # The expiring operator lease also stops the controller.
        monitor.stop_event.set()
        worker.join(timeout=10)
        root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Cámara y YOLO del tractor; parada opcional en simulación"
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Abre Gazebo en WSL y el visor con GPU de Windows",
    )
    parser.add_argument("--distribution", default="Ubuntu-24.04")
    parser.add_argument(
        "--vision-brake",
        action="store_true",
        help="Frena con Person/Dog en la cámara Gazebo de esta sesión; nunca con video grabado",
    )
    parser.add_argument(
        "--frames-dir", type=Path, default=PROJECT / "runs/tractor_camera/live"
    )
    parser.add_argument(
        "--weights", type=Path, default=PROJECT / "models/translate/best.pt"
    )
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument(
        "--source", help="Video real o índice de webcam (0); no inicia Gazebo"
    )
    parser.add_argument(
        "--select-source",
        action="store_true",
        help="Elegir una fotografía o un video real en una ventana",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0,
        help="Empieza el video en este segundo",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Carpeta para los resultados de esta prueba"
    )
    parser.add_argument(
        "--timeout", type=float, default=0, help="Segundos reales; 0 sin límite"
    )
    args = parser.parse_args()
    if args.select_source:
        if args.launch or args.source is not None:
            parser.error("--select-source se usa sin --launch ni --source")
        import tkinter as tk
        from tkinter import filedialog

        picker = tk.Tk()
        picker.withdraw()
        try:
            selected = filedialog.askopenfilename(
                parent=picker,
                title="Selecciona una foto o un video para YOLO",
                filetypes=[
                    (
                        "Fotos y videos",
                        "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mov *.mkv",
                    ),
                    ("Todos los archivos", "*.*"),
                ],
            )
        finally:
            picker.destroy()
        if not selected:
            return 0
        args.source = selected
    if (
        not 0 < args.confidence <= 1
        or args.stride < 1
        or args.max_frames < 0
        or args.timeout < 0
        or not np.isfinite(args.start_seconds)
        or args.start_seconds < 0
    ):
        parser.error("Confianza, stride, máximo de fotogramas o timeout inválidos")
    if args.start_seconds and (args.source is None or args.source.isdecimal()):
        parser.error("--start-seconds requiere un archivo de video en --source")
    if args.launch and args.source is not None:
        parser.error("Usa --launch para Gazebo o --source para video/cámara real")
    if args.vision_brake and (args.source is not None or args.output_dir is not None):
        parser.error(
            "--vision-brake usa solo la cámara Gazebo y su --frames-dir; sin --source ni --output-dir"
        )
    simulation = None
    try:
        # Verify weights/dependencies before opening a simulation.
        monitor = VisionMonitor(args)
        if args.launch:
            simulation = launch_gazebo(args)
            monitor.output = args.frames_dir
        if args.headless:
            monitor.run()
        else:
            show_window(monitor, simulation)
        if monitor.error:
            print(monitor.error, file=sys.stderr)
            return 1
        return 0 if monitor.processed else 1
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"No se pudo iniciar YOLO: {error}", file=sys.stderr)
        return 1
    finally:
        if simulation is not None:
            (args.frames_dir / "stop.request").write_text("close\n", encoding="utf-8")
            try:
                simulation.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print(
                    f"La sesión sigue cerrándose. Revisa {args.frames_dir / 'gazebo.log'}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
