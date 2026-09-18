import json
from datetime import datetime
from pathlib import Path

import av
import cv2
import pandas as pd
import streamlit as st

from cctv_incident.calibration import estimate_calibration, load_calibration
from cctv_incident.calibration.assisted import propose_calibration
from cctv_incident.calibration.background import sample_background
from cctv_incident.config import load_config
from cctv_incident.pipeline import Pipeline
from cctv_incident.replay import export_replay, list_saved_runs
from cctv_incident.storage import EventStorage

st.set_page_config(page_title="CCTV Incident Detection", page_icon="🚦", layout="wide")
st.title("CCTV · Incident Detection")
st.caption(
    "Analisi locale di una telecamera · Traiettorie, geometria stradale e conferma temporale"
)
preset = st.sidebar.selectbox(
    "Modalita",
    ["Demo sintetica", "Video reale senza calibrazione", "Video con calibrazione metrica"],
)
default_configs = {
    "Demo sintetica": "configs/demo.yaml",
    "Video reale senza calibrazione": "configs/accident-image.yaml",
    "Video con calibrazione metrica": "configs/default.yaml",
}
config_path = st.sidebar.text_input(
    "Configurazione YAML", default_configs[preset], key=f"config_{preset}"
)
try:
    cfg = load_config(config_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()
if cfg.events.coordinate_mode == "image":
    st.info(
        "Analisi in coordinate immagine: moto in px/s e soglie relative ai veicoli. "
        "La precisione del rilevamento deve essere verificata su dati reali."
    )
    manifest = cfg.project.root_dir / "outputs/accident-sample/sample.json"
    if manifest.is_file():
        videos = json.loads(manifest.read_text(encoding="utf-8"))["videos"]
        selected_clip = st.sidebar.selectbox(
            "Video del campione ACCIDENT",
            ["Percorso personalizzato"] + [r["clip_id"] for r in videos],
        )
        if selected_clip != "Percorso personalizzato":
            selected_row = next(r for r in videos if r["clip_id"] == selected_clip)
            cfg.video.source = str(
                cfg.project.root_dir / "data/raw/ACCIDENT" / selected_row["path"]
            )
            cfg.video.clip_id = selected_clip
            cfg.calibration.camera_id = selected_row["camera_id"]
if cfg.events.coordinate_mode == "image" and cfg.perception.backend == "pytorch":
    analysis_quality = st.sidebar.selectbox(
        "Qualità analisi",
        ["Rapida", "Accurata"],
        help="Accurata usa un modello più capace e analizza più fotogrammi; richiede più tempo su CPU.",
    )
    if analysis_quality == "Accurata":
        accurate_model = cfg.project.root_dir / "models/yolo26m-seg.pt"
        if not accurate_model.is_file():
            st.error("Il modello per l'analisi accurata non è installato.")
            st.stop()
        cfg.perception.model = accurate_model
        cfg.perception.image_size = 640
        cfg.video.target_fps = 15
source = st.sidebar.text_input(
    "Percorso video o URL RTSP", cfg.video.source, key=f"source_{cfg.video.source}"
)
cfg.video.source = source
st.sidebar.caption(f"Backend: {cfg.perception.backend} · {cfg.video.target_fps:g} FPS richiesti")
if cfg.perception.backend == "synthetic":
    st.warning(
        "Modalità sintetica: le maschere sono ricavate dai colori della demo. Per CCTV selezionare configs/default.yaml."
    )
tab_analysis, tab_replay, tab_calibration, tab_events = st.tabs(
    ["Analisi", "Rivedi analisi", "Calibrazione", "Eventi"]
)

with tab_analysis:
    if st.button("Genera video dimostrativi"):
        from cctv_incident.demo import generate_demo

        generate_demo(cfg.project.root_dir)
        generate_demo(cfg.project.root_dir, negative=True)
        st.success("Video positivo, negativo e calibrazione sintetica creati.")
    columns = st.columns([2, 1])
    video_slot = columns[0].empty()
    bird_slot = columns[1].empty()
    status_slot = st.empty()
    metric_slot = st.empty()
    # Streamlit reruns cancel the current script; Pipeline.finally flushes clips.
    if st.button("Avvia analisi", type="primary"):
        try:

            def update(data):
                video_slot.image(data["frame"], channels="BGR")
                if data["bird_eye"] is not None:
                    bird_slot.image(data["bird_eye"], channels="BGR")
                decision = data["decision"]
                status_slot.info(
                    f"{decision.state} · Punteggio {decision.score:.2f} · "
                    + ", ".join(decision.reasons)
                )
                metric_slot.caption(
                    f"Tempo video {data['timestamp_s']:.2f} s · FPS effettivi {data['fps']:.1f}"
                )

            summary = Pipeline(cfg).run(update)
            st.session_state["last_run"] = summary
            st.success(f"Analisi completata: {summary['events']} eventi.")
            if not source.lower().startswith(("rtsp://", "rtsps://")):
                try:
                    with st.spinner("Preparazione del video annotato..."):
                        export_replay(summary["run_dir"])
                    st.success("Video pronto nella scheda Rivedi analisi.")
                except Exception as exc:
                    st.warning(f"Analisi salvata, ma il video annotato non è pronto: {exc}")
        except Exception as exc:
            st.error(f"Analisi interrotta: {exc}")
    if "last_run" in st.session_state:
        st.json(st.session_state["last_run"], expanded=False)


with tab_replay:
    saved_runs = list_saved_runs(cfg.project.output_dir)
    if not saved_runs:
        st.info("Avvia l'analisi di un video per rivederlo con le annotazioni.")
    else:
        runs_by_id = {run["run_id"]: run for run in saved_runs}

        def run_label(run_id):
            run = runs_by_id[run_id]
            date = datetime.fromtimestamp(run["modified_at"]).strftime("%d/%m/%Y %H:%M")
            return f"{run['source_name']} · {date} · {run['events']} eventi"

        selected_run = st.selectbox("Analisi da rivedere", list(runs_by_id), format_func=run_label)
        run_dir = Path(runs_by_id[selected_run]["run_dir"])
        replay_path = run_dir / "annotated.mp4"
        replay_report = run_dir / "replay.json"
        if not replay_path.is_file() or not replay_report.is_file():
            st.info(
                "Questa analisi è salvata. Puoi preparare il video senza ripetere il rilevamento."
            )
            if st.button("Prepara video annotato", key=f"prepare_{selected_run}"):
                try:
                    with st.spinner("Preparazione del video annotato..."):
                        export_replay(run_dir)
                except Exception as exc:
                    st.error(f"Impossibile preparare il video: {exc}")
        if replay_path.is_file() and replay_report.is_file():
            replay = json.loads(replay_report.read_text(encoding="utf-8"))
            impacts = replay["impacts"]
            start_time = 0.0
            if impacts:
                options = ["Video completo"] + [
                    f"Impatto {index} · {marker['impact_time_s']:.2f} s · "
                    + "ID "
                    + ", ".join(map(str, marker["track_ids"]))
                    for index, marker in enumerate(impacts, 1)
                ]
                selected_impact = st.selectbox(
                    "Vai a un impatto",
                    range(len(options)),
                    format_func=lambda index: options[index],
                    key=f"impact_{selected_run}",
                )
                if selected_impact:
                    marker = impacts[selected_impact - 1]
                    start_time = max(0.0, marker["impact_time_s"] - 1)
                st.caption(
                    "Il segnale rosso indica il punto stimato dell'impatto rilevato, "
                    "a partire dal tempo stimato dell'urto. La posizione è ricavata "
                    "dalle bounding box dei veicoli coinvolti."
                )
                if any(marker["point_px"] is None for marker in impacts):
                    st.info(
                        "Per alcuni eventi la posizione non è disponibile: è indicato solo il tempo."
                    )
            else:
                st.info("Nessun impatto rilevato in questa analisi.")
            st.video(str(replay_path), start_time=start_time)
            st.caption(
                "Bounding box, ID, traiettorie e velocità dei veicoli. Usa la barra del video per rivedere qualsiasi momento."
            )
            with replay_path.open("rb") as video_file:
                st.download_button(
                    "Scarica video annotato",
                    video_file,
                    file_name=f"{Path(replay['source_name']).stem}_annotato.mp4",
                    mime="video/mp4",
                    key=f"download_{selected_run}",
                )

with tab_calibration:
    st.write(
        "Inserire almeno quattro corrispondenze sul piano stradale, nello stesso ordine nelle due liste. Le coordinate metriche devono derivare da misure reali."
    )
    if st.button("Estrai primo frame"):
        try:
            with av.open(source) as container:
                frame = next(container.decode(video=0)).to_ndarray(format="bgr24")
                st.session_state["calibration_frame"] = frame
        except Exception as exc:
            st.error(str(exc))
    if st.button("Calcola sfondo mediano"):
        try:
            background, unstable, original_size = sample_background(source)
            st.session_state["calibration_frame"] = cv2.resize(background, original_size)
            st.image(unstable, caption="Aree instabili", clamp=True)
        except Exception as exc:
            st.error(str(exc))
    frame = st.session_state.get("calibration_frame")
    if frame is not None:
        st.image(
            frame,
            channels="BGR",
            caption=f"Coordinate in pixel: {frame.shape[1]} × {frame.shape[0]}",
        )
    source_text = st.text_area(
        "Punti immagine [x, y] (JSON)", "[[100,100],[500,100],[500,300],[100,300]]"
    )
    destination_text = st.text_area(
        "Punti sul piano stradale [x, y] (JSON)", "[[0,0],[10,0],[10,20],[0,20]]"
    )
    metric = st.checkbox("Coordinate misurate in metri", value=False)
    target = st.text_input("File calibrazione", str(cfg.calibration.file))
    if st.button("Salva calibrazione manuale"):
        try:
            if frame is None:
                raise ValueError("Estrarre prima un frame del video")
            calibration = estimate_calibration(
                cfg.calibration.camera_id,
                (frame.shape[1], frame.shape[0]),
                json.loads(source_text),
                json.loads(destination_text),
                units="m" if metric else "canonical",
            )
            reference_path = Path(target).with_suffix(".reference.jpg")
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(reference_path), frame)
            calibration.reference_image = reference_path.name
            calibration.save(target)
            st.success("Calibrazione salvata.")
        except Exception as exc:
            st.error(str(exc))
    st.write(
        "La proposta automatica individua intersezioni delle strisce. Verificare i vertici e fornire misure prima di usarla per il detector."
    )
    if st.button("Proponi calibrazione dalle strisce"):
        if frame is None:
            st.error("Estrarre prima un frame.")
        else:
            proposal, diagnostics, mask, lines = propose_calibration(
                frame, cfg.calibration.camera_id
            )
            st.image([mask, cv2.cvtColor(lines, cv2.COLOR_BGR2RGB)], caption=["Strisce", "Linee"])
            st.json(diagnostics)
            if proposal:
                st.session_state["proposal"] = proposal.model_dump()
    if "proposal" in st.session_state:
        st.code(json.dumps(st.session_state["proposal"]["source_points_px"]))
        st.caption(
            "Copiare o correggere questi punti nel modulo manuale e assegnare coordinate misurate."
        )
    if Path(target).exists():
        st.json(load_calibration(target).model_dump(), expanded=False)

with tab_events:
    storage = EventStorage(cfg.project.output_dir)
    try:
        events = storage.list_events()
    finally:
        storage.close()
    if events:
        table = pd.DataFrame(events)
        st.dataframe(table.drop(columns=["reasons", "clip_path"]), hide_index=True)
        st.scatter_chart(table, x="impact_time_s", y="score", color="camera_id")
        selected = st.selectbox("Evento", [row["event_id"] for row in events])
        event = next(row for row in events if row["event_id"] == selected)
        st.write("Motivi: " + ", ".join(event["reasons"]))
        if event["clip_path"] and Path(event["clip_path"]).exists():
            st.video(event["clip_path"])
        else:
            st.caption(f"Clip: {event['clip_status']}")
        st.download_button(
            "Esporta JSON", json.dumps(events, indent=2), "events.json", "application/json"
        )
        st.download_button("Esporta CSV", table.to_csv(index=False), "events.csv", "text/csv")
    else:
        st.info("Nessun evento registrato.")
