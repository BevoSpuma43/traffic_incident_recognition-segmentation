"""Streamlit controls for durable batch workers; no inference in the UI thread."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from .batch import (
    list_jobs,
    prepare_job,
    read_json,
    snapshot,
    start_job,
    stop_job,
    worker_alive,
)


def render_batch_page(cfg):
    root = cfg.project.root_dir
    output_root = root / "outputs/batches"
    st.subheader("Analisi sequenziale della cartella")
    st.write(
        "Ogni video viene confrontato con metadata-real.csv. "
        "Gli eventi corrispondono se distano al massimo ±1 secondo dall'incidente annotato."
    )
    st.caption(
        "Stop interrompe dopo il fotogramma in corso e il salvataggio delle clip. "
        "Riprendi salta i video completati e ricomincia dall'inizio del video interrotto. "
        "Chiudere la pagina non arresta il processo: usa Stop prima di spegnere il computer."
    )
    models = sorted((root / "models").rglob("*.pt"))
    if not models:
        st.error("Nessun modello .pt disponibile nella cartella models.")
        return
    model = st.sidebar.selectbox(
        "Modello YOLO",
        models,
        format_func=lambda p: p.relative_to(root / "models").as_posix(),
        index=next((i for i, p in enumerate(models) if p.name == "yolo26s-seg.pt"), 0),
        key="batch_model",
        help="Servono pesi di segmentazione. Ogni modello conserva risultati e checkpoint propri.",
    )
    dataset = root / "dataset"
    directories = (
        sorted(p.name for p in dataset.iterdir() if p.is_dir()) if dataset.is_dir() else []
    )
    choices = directories + ["Percorso personalizzato"]
    selected = st.sidebar.selectbox(
        "Cartella nel dataset",
        choices,
        index=choices.index("standard_dataset")
        if "standard_dataset" in choices
        else len(choices) - 1,
    )
    folder = st.sidebar.text_input(
        "Cartella video",
        str(dataset / selected) if selected in directories else str(dataset),
        key=f"batch_folder_{selected}",
    )
    metadata = st.sidebar.text_input("CSV delle etichette", str(dataset / "metadata-real.csv"))
    fps = st.sidebar.number_input("FPS da analizzare", min_value=1.0, max_value=60.0, value=8.0)
    st.sidebar.caption("Mantieni gli stessi FPS e parametri per confrontare i modelli.")
    cfg = cfg.model_copy(deep=True)
    cfg.perception.model = model
    cfg.perception.backend = "pytorch"
    cfg.video.target_fps = fps
    if st.button("Prepara batch", type="primary"):
        try:
            with st.spinner("Verifica video, etichette e modello..."):
                job = prepare_job(cfg, folder, metadata, output_root, tolerance_s=1.0)
            st.session_state["batch_selected_job"] = str(job)
            st.success("Batch pronto. Premi Avvia batch oppure Riprendi se esiste un checkpoint.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))

    jobs = list_jobs(output_root)
    matching = [item for item in jobs if Path(item["model_path"]) == model]
    if matching:
        options = {item["path"]: item for item in matching}
        if st.session_state.get("batch_selected_job") not in options:
            st.session_state["batch_selected_job"] = matching[0]["path"]

        def job_label(path):
            item = options[path]
            date = datetime.fromtimestamp(item["created_at"]).strftime("%d/%m/%Y %H:%M")
            return f"{Path(item['folder']).name} · {item['total_videos']} video · {date} · {Path(path).name[-16:]}"

        selected_job = st.selectbox(
            "Esperimento salvato per questo modello",
            list(options),
            format_func=job_label,
            key="batch_selected_job",
        )
        manifest = read_json(Path(selected_job) / "manifest.json")
        st.caption(
            f"Esperimento selezionato: {manifest['folder']} · "
            f"{manifest['config']['video']['target_fps']:g} FPS · "
            f"tolleranza ±{manifest['tolerance_s']:g} s. "
            "Le impostazioni laterali valgono per Prepara batch; Riprendi usa quelle salvate."
        )
        if all(v["label"]["positive"] for v in manifest["videos"]):
            st.info(
                "Tutti i video hanno un incidente annotato: nel report per video TN e FP saranno 0. "
                "Un allarme fuori tempo è un FP nel report eventi. "
                "TN e accuracy non sono definiti per gli eventi; le celle non definite restano vuote."
            )
    else:
        selected_job = None
        st.info("Prepara la cartella per creare il primo esperimento di questo modello.")
    batch_monitor(selected_job, str(output_root))


@st.fragment(run_every=1.0)
def batch_monitor(selected_job, output_root):
    jobs = list_jobs(output_root)
    active = [item for item in jobs if worker_alive(item["path"])]
    for item in active:
        if item["path"] != selected_job:
            st.warning(f"Batch attivo con {item['model_name']}: {Path(item['folder']).name}")
            if st.button("Stop batch attivo", key=f"stop_{Path(item['path']).name}"):
                stop_job(item["path"])
                st.info("Arresto richiesto. Attendi la fine del fotogramma e il salvataggio.")
    if selected_job is None:
        return
    job = Path(selected_job)
    state = snapshot(job)
    finished = state["status"] == "completed"
    busy = bool(active)
    buttons = st.columns(3)
    begin = buttons[0].button(
        "Avvia batch", disabled=busy or finished or state["status"] != "ready"
    )
    resume = buttons[1].button("Riprendi", disabled=busy or finished or state["status"] == "ready")
    stop = buttons[2].button("Stop", disabled=not state["active"] or state["stop_requested"])
    if begin or resume:
        try:
            start_job(job)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
        state = snapshot(job)
    if stop:
        stop_job(job)
        state = snapshot(job)
    statuses = {
        "ready": "Pronto",
        "starting": "Caricamento",
        "running": "In analisi",
        "paused": "In pausa",
        "interrupted": "Interrotto: checkpoint recuperabile",
        "error": "Errore: ripresa disponibile",
        "completed": "Completato",
    }
    status = (
        "Arresto in corso"
        if state["active"] and state["stop_requested"]
        else statuses[state["status"]]
    )
    st.write(f"**{status}** · {state['completed_videos']}/{state['total_videos']} video completati")
    st.progress(
        state["completed_videos"] / state["total_videos"], text="Avanzamento della cartella"
    )
    if state.get("current_video"):
        percent = state.get("progress", 0.0)
        st.write(
            f"Video {state['current_index']}/{state['total_videos']}: **{state['current_video']}**"
        )
        st.progress(percent, text=f"Video corrente: {percent:.1%}")
        st.caption(f"Tempo video elaborato: {state.get('timestamp_s', 0):.2f} s")
    if state.get("error"):
        st.error(state["error"])
    metrics = read_json(job / "metrics.json")
    st.write(
        "**Metriche definitive**"
        if metrics["complete"]
        else "**Metriche parziali sui video completati**"
    )
    rows = []
    for unit, title in (("video", "Per video"), ("event", "Eventi (±1 s)")):
        rows.append({"Valutazione": title, **metrics[unit]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Risultati e checkpoint: {job}")
    for filename, title in (
        ("videos.csv", "Scarica risultati per video"),
        ("events.csv", "Scarica incidenti rilevati"),
        ("metrics.csv", "Scarica metriche"),
    ):
        st.download_button(
            title,
            (job / filename).read_bytes(),
            file_name=f"{job.name}-{filename}",
            mime="text/csv",
            key=f"download_batch_{filename}",
        )
    if st.checkbox("Mostra confronto tra esperimenti", key="batch_comparison"):
        comparison = []
        for item in jobs:
            report = read_json(Path(item["path"]) / "metrics.json")
            comparison.extend(
                {
                    "Esperimento": Path(item["path"]).name,
                    "Modello": item["model_name"],
                    "Cartella": item["folder"],
                    "Video completati": report["completed_videos"],
                    "Video totali": report["total_videos"],
                    "Completo": report["complete"],
                    "Unità": unit,
                    **report[unit],
                }
                for unit in ("video", "event")
            )
        st.dataframe(pd.DataFrame(comparison), hide_index=True)
        st.caption("Confronta esperimenti completi con gli stessi video, FPS e protocollo.")
