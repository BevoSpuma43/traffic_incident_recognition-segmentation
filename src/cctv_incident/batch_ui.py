"""Streamlit controls for durable batch workers; no inference in the UI thread."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from .batch import (
    list_jobs,
    prepare_job,
    read_json,
    resume_compatibility,
    snapshot,
    start_job,
    stop_job,
    worker_alive,
)
from .batch_selection import active_selection, load_selection
from .calibration.preparation import analysis_readiness, prepared_archive_mapping


def render_batch_page(cfg, *, mode, config_path):
    metric = cfg.events.coordinate_mode == "metric"
    prefix = "metric_" if metric else ""
    selection_key = prefix + "batch_selected_job"
    root = cfg.project.root_dir
    try:
        selection_path = active_selection(root)
        sample = load_selection(selection_path) if selection_path else None
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Campione condiviso non disponibile: {exc}")
        return
    preparation_job = st.session_state.get("metric_preparation_job") if metric else None
    preparation = None
    if preparation_job:
        try:
            preparation = read_json(Path(preparation_job) / "manifest.json")
        except (ValueError, OSError):
            preparation_job = None
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
    jobs = list_jobs(output_root)
    models = sorted(
        set((root / "models").rglob("*.pt"))
        | {
            Path(item["model_path"])
            for item in jobs
            if item["coordinate_mode"] == cfg.events.coordinate_mode
        }
    )
    if not models:
        st.error("Nessun modello .pt disponibile nella cartella models.")
        return
    model = st.sidebar.selectbox(
        "Modello YOLO",
        models,
        format_func=lambda p: (
            (
                p.relative_to(root / "models").as_posix()
                if p.is_relative_to(root / "models")
                else str(p)
            )
            + (" (pesi non disponibili)" if not p.is_file() else "")
        ),
        index=next((i for i, p in enumerate(models) if p.name == "yolo26s-seg.pt"), 0),
        key=prefix + "batch_model",
        help="Servono pesi di segmentazione. Ogni modello conserva risultati e checkpoint propri.",
    )
    dataset = root / "dataset"
    directories = (
        sorted(p.name for p in dataset.iterdir() if p.is_dir()) if dataset.is_dir() else []
    )
    if sample:
        selected, folder = "Campione condiviso", sample["folder"]
        st.success(
            f"Campione condiviso: {len(sample['videos'])}/{sample['original_count']} video. "
            "Sia l'omografia sia la sola segmentazione useranno esattamente questo elenco."
        )
        st.sidebar.caption(f"Cartella del campione: {folder}")
        st.caption(f"Campione: {sample['selection_id']} · {len(sample['excluded'])} video esclusi.")
        st.dataframe(
            [{"Video selezionato": v["relative_path"]} for v in sample["videos"]], hide_index=True
        )
        st.download_button(
            "Scarica elenco del campione",
            json.dumps(
                {
                    "campione": sample["selection_id"],
                    "video": [v["relative_path"] for v in sample["videos"]],
                },
                indent=2,
            ),
            file_name=sample["selection_id"] + ".json",
            mime="application/json",
            key=prefix + "sample_download",
        )
    elif metric:
        selected = "Preparazione calibrazioni"
        folder = preparation["folder"] if preparation else str(dataset)
        st.sidebar.caption(
            f"Cartella della preparazione: {folder}"
            if preparation
            else "Seleziona e completa una sessione in Preparazione calibrazioni."
        )
    else:
        choices = directories + ["Percorso personalizzato"]
        selected = st.sidebar.selectbox(
            "Cartella nel dataset",
            choices,
            index=choices.index("standard_dataset")
            if "standard_dataset" in choices
            else len(choices) - 1,
            key=prefix + "batch_dataset",
        )
        folder = st.sidebar.text_input(
            "Cartella video",
            str(dataset / selected) if selected in directories else str(dataset),
            key=f"{prefix}batch_folder_{selected}",
        )
    readiness = None
    if metric and preparation_job and not sample:
        try:
            readiness = analysis_readiness(
                preparation_job, min_quality=cfg.calibration.min_confidence
            )
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Preparazione non disponibile: {exc}")
    if readiness:
        st.write("**Riepilogo calibrazioni prima dell'analisi**")
        st.dataframe(
            [
                {
                    "Video": r["video"],
                    "Stato": "Utilizzabile"
                    if r["state"] == "confirmed" and (r["quality"] or 0) >= readiness["min_quality"]
                    else "Da correggere",
                    "Accettazione": "Automatica sperimentale"
                    if r["acceptance"] == "automatic"
                    else ("Manuale" if r["acceptance"] else "—"),
                    "Riferimento": r["reference_type"],
                    "Larghezza (m)": r["width_m"],
                    "Lunghezza (m)": r["length_m"],
                    "Motivo": r["message"],
                }
                for r in readiness["rows"]
            ],
            hide_index=True,
        )
    metadata = st.sidebar.text_input(
        "CSV delle etichette", str(dataset / "metadata-real.csv"), key=prefix + "batch_metadata"
    )
    fps = st.sidebar.number_input(
        "FPS da analizzare", min_value=1.0, max_value=60.0, value=8.0, key=prefix + "batch_fps"
    )
    st.sidebar.caption("Mantieni gli stessi FPS e parametri per confrontare i modelli.")
    st.caption(
        "L'avvio crea un nuovo esperimento in una cartella separata. "
        "Per continuare un esperimento esistente, selezionalo sotto e usa Riprendi."
    )
    cfg = cfg.model_copy(deep=True)
    cfg.perception.model = model
    cfg.perception.backend = "pytorch"
    cfg.video.target_fps = fps
    if metric:
        if sample:
            cfg.calibration.min_confidence = sample["min_quality"]
        elif readiness:
            cfg.calibration.min_confidence = readiness["min_quality"]
        st.info(
            "L'analisi usa le calibrazioni del riepilogo e ne conserva copie indipendenti. "
            "Seleziona un campione di video calibrati oppure completa tutte le correzioni manuali. "
            "Le calibrazioni automatiche sperimentali producono misure approssimative."
        )
    if st.button(
        "Avvia analisi con queste calibrazioni" if metric else "Prepara batch",
        type="primary",
        key=prefix + "batch_prepare",
        disabled=not model.is_file()
        or (metric and not sample and not (readiness and readiness["ready"])),
    ):
        try:
            with st.spinner("Verifica video, etichette e modello..."):
                calibration_map = (
                    prepared_archive_mapping(
                        preparation_job, folder, min_quality=cfg.calibration.min_confidence
                    )
                    if metric and not sample
                    else None
                )
                job = prepare_job(
                    cfg,
                    folder,
                    metadata,
                    output_root,
                    tolerance_s=1.0,
                    mode=mode,
                    config_path=config_path,
                    dataset_directory=dataset,
                    dataset_selection=selected,
                    calibration_map=calibration_map,
                    selection_path=selection_path,
                )
            st.session_state[selection_key] = str(job)
            if metric:
                start_job(job)
                st.success("Analisi avviata con le calibrazioni del riepilogo.")
            else:
                st.success("Nuovo batch pronto: configurazione salvata. Premi Avvia batch.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))

    jobs = list_jobs(output_root)
    matching = [
        item
        for item in jobs
        if Path(item["model_path"]) == model
        and item["coordinate_mode"] == cfg.events.coordinate_mode
        and (not sample or item.get("selection_id") == sample["selection_id"])
    ]
    if matching:
        options = {item["path"]: item for item in matching}
        if st.session_state.get(selection_key) not in options:
            st.session_state[selection_key] = matching[0]["path"]

        def job_label(path):
            item = options[path]
            date = datetime.fromtimestamp(item["created_at"]).strftime("%d/%m/%Y %H:%M")
            return f"{Path(item['folder']).name} · {item['total_videos']} video · {date} · {Path(path).name[-16:]}"

        selected_job = st.selectbox(
            "Esperimento salvato per questo modello",
            list(options),
            format_func=job_label,
            key=selection_key,
        )
        manifest = read_json(Path(selected_job) / "manifest.json")
        if "batch_settings" in manifest:
            with st.expander("Configurazione salvata dell'esperimento"):
                settings = manifest["batch_settings"]
                st.json(
                    {
                        key: value
                        for key, value in settings.items()
                        if key != "resolved_configuration"
                    }
                )
        st.caption(
            f"Esperimento selezionato: {manifest['folder']} · "
            f"{manifest['config']['video']['target_fps']:g} FPS · "
            f"tolleranza ±{manifest['tolerance_s']:g} s. "
            "Le impostazioni laterali valgono per i nuovi esperimenti; Riprendi usa quelle salvate."
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
    batch_monitor(selected_job, str(output_root), prefix)


@st.fragment(run_every=1.0)
def batch_monitor(selected_job, output_root, prefix=""):
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
    invalid = state.get("requires_new_experiment", False)
    compatible, compatibility_message = resume_compatibility(job)
    if not compatible:
        st.warning(compatibility_message)
    widget_prefix = prefix + job.name + "_"
    busy = bool(active)
    buttons = st.columns(3)
    begin = buttons[0].button(
        "Avvia batch",
        disabled=busy or finished or invalid or not compatible or state["status"] != "ready",
        key=widget_prefix + "start",
    )
    resume = buttons[1].button(
        "Riprendi",
        disabled=busy or finished or invalid or not compatible or state["status"] == "ready",
        key=widget_prefix + "resume",
    )
    stop = buttons[2].button(
        "Stop", disabled=not state["active"] or state["stop_requested"], key=widget_prefix + "stop"
    )
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
    if invalid:
        status = "Calibrazione invalidata: serve un nuovo esperimento"
    elif not compatible and not state["active"] and not finished:
        status = "Consultabile; ripresa con il codice originale"
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
            key=f"{widget_prefix}download_batch_{filename}",
        )
    if st.checkbox("Mostra confronto tra esperimenti", key=prefix + "batch_comparison"):
        comparison = []
        for item in jobs:
            report = read_json(Path(item["path"]) / "metrics.json")
            comparison.extend(
                {
                    "Esperimento": Path(item["path"]).name,
                    "Modello": item["model_name"],
                    "Modalità": item["coordinate_mode"] or "Non registrata",
                    "Cartella": item["folder"],
                    "Campione": item.get("selection_id") or "Cartella completa",
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
