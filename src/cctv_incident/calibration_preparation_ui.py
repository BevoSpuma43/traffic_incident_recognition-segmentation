"""Preparation and review, separate from metric batch inference."""

from datetime import datetime
from pathlib import Path

import streamlit as st

from .batch import read_json, write_json
from .batch_selection import create_automatic_selection
from .calibration.preparation import (
    analysis_readiness,
    list_sessions,
    load_proposal,
    prepare_session,
    snapshot,
    start_preparation,
    stop_preparation,
    table_rows,
)
from .calibration_ui import ORIGINS, render_calibration_editor

STATES = {
    "missing": "Mancante",
    "draft": "Bozza",
    "confirmed": "Confermato",
    "incompatible": "Incompatibile",
    "error": "Errore",
}
PREPARATION = {
    "auto_calibrated": "Calibrato automaticamente (approssimativo)",
    "manual_required": "Da calibrare a mano",
    "pending": "Da preparare",
    "proposed": "Proposta salvata",
    "no_reference": "Riferimento non trovato",
    "kept": "Record conservato",
    "incompatible": "Da correggere",
    "error": "Errore di preparazione",
}


@st.fragment(run_every=1.0)
def preparation_monitor(job):
    state = snapshot(job)
    automatic = read_json(Path(job) / "manifest.json").get("automatic_parameters") is not None
    key = "preparation_active_" + Path(job).name
    previously_active = st.session_state.get(key, state["active"])
    st.session_state[key] = state["active"]
    if previously_active and not state["active"]:
        st.rerun()
    st.progress(
        state["completed_videos"] / state["total_videos"],
        text=f"{state['completed_videos']}/{state['total_videos']} video esaminati",
    )
    labels = {
        "ready": "Pronta",
        "starting": "Avvio",
        "running": "Preparazione in corso",
        "paused": "In pausa",
        "interrupted": "Interrotta: ripresa disponibile",
        "completed": "Preparazione terminata: controlla il riepilogo",
        "error": "Errore",
    }
    st.write(labels[state["status"]])
    if state.get("current_video"):
        st.caption(f"Video corrente: {state['current_video']}")
    if state.get("error"):
        st.error(state["error"])
    buttons = st.columns(3)
    finished = state["completed_videos"] == state["total_videos"]
    start = buttons[0].button(
        "Calibra tutti i video" if automatic else "Proponi calibrazioni mancanti",
        disabled=state["active"] or finished or state["status"] != "ready",
        key="prep_start",
    )
    resume = buttons[1].button(
        "Riprendi proposte",
        disabled=state["active"] or finished or state["status"] == "ready",
        key="prep_resume",
    )
    stop = buttons[2].button(
        "Stop proposte", disabled=not state["active"] or state["stop_requested"], key="prep_stop"
    )
    if start or resume:
        try:
            with st.spinner("Verifica degli input e avvio del worker..."):
                start_preparation(job)
            # Drop stale editor drafts before the worker can publish an archive record.
            for name in list(st.session_state):
                if name.startswith("prep_") and name.endswith("_calibration_editor"):
                    del st.session_state[name]
            st.rerun()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if stop:
        stop_preparation(job)
        st.rerun()


def render_preparation_page(cfg):
    root = Path(cfg.project.root_dir).resolve()
    output = root / "outputs/calibration-preparations"
    st.subheader("Preparazione calibrazioni per il batch metrico")
    st.info(
        "Prepara le calibrazioni di tutti i video, controlla il riepilogo e correggi a mano i casi falliti. "
        "Poi passa all'analisi e premi il pulsante di avvio. Ogni esperimento conserva una copia delle calibrazioni."
    )
    st.caption(
        "Nuove sessioni: USA ipotizzati, modalità sperimentale, primi 10 secondi. "
        "Tratti 3,048 × 0,1524 m; barre pedonali 0,4572 × 3 m; corsia 3,6576 m; "
        "auto tipo 4,7 × 1,8 × 1,5 m. Misure approssimative: barra lunga 3 m e auto tipo "
        "sono scelte del progetto. Chiudere la pagina non ferma il worker: usa Stop proposte."
    )
    folder = st.text_input(
        "Cartella video da calibrare", str(root / "dataset/standard_dataset"), key="prep_folder"
    )
    models = sorted((root / "models").rglob("*seg*.pt"))
    model = (
        st.selectbox(
            "Modello locale per la calibrazione dalle auto",
            models,
            index=next((i for i, p in enumerate(models) if p.name == "yolo26s-seg.pt"), 0),
            format_func=lambda p: p.name,
            key="prep_model",
        )
        if models
        else None
    )
    if not models:
        st.caption(
            "Nessun peso di segmentazione locale: saranno cercati solo riferimenti stradali."
        )
    if st.button("Crea sessione di preparazione", key="prep_create"):
        try:
            with st.spinner("Lettura dei primi fotogrammi e degli hash..."):
                job = prepare_session(
                    root,
                    folder,
                    automatic_parameters={},
                    model_path=model,
                    parameters={"max_candidates": 8},
                )
            st.session_state["prep_session"] = str(job)
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
    sessions = list_sessions(output)
    if not sessions:
        st.info("Crea una sessione per elencare e calibrare i video della cartella.")
        return
    options = {item["path"]: item for item in sessions}
    if st.session_state.get("prep_session") not in options:
        st.session_state["prep_session"] = sessions[0]["path"]

    def label(path):
        item = options[path]
        date = datetime.fromtimestamp(item["created_at"]).strftime("%d/%m/%Y %H:%M")
        return (
            f"{Path(item['folder']).name} · {item['total']} video · {date} · {Path(path).name[-8:]}"
        )

    job = Path(
        st.selectbox(
            "Sessione di preparazione salvata", list(options), format_func=label, key="prep_session"
        )
    )
    manifest = read_json(job / "manifest.json")
    st.session_state["metric_preparation_job"] = str(job)
    st.caption(
        f"Sessione selezionata: {manifest['folder']}. La cartella sopra vale per una nuova sessione."
    )
    preparation_monitor(job)
    refresh = st.button("Aggiorna stato e verifica video", key="prep_refresh")
    try:
        rows = table_rows(job, verify_sources=refresh)
    except (OSError, ValueError) as exc:
        st.error(f"Impossibile leggere la preparazione: {exc}")
        return
    select_key = "prep_video_" + job.name
    cursor_path = job / "review.json"
    if select_key not in st.session_state:
        cursor = read_json(cursor_path).get("index", 0) if cursor_path.is_file() else 0
        st.session_state[select_key] = cursor if 0 <= cursor < len(rows) else 0
    display = [
        {
            "Video": r["video"],
            "Calibrazione": STATES[r["state"]],
            "Accettazione": "Automatica sperimentale"
            if r["acceptance"] == "automatic"
            else ("Manuale" if r["acceptance"] else "—"),
            "Riferimento": r["reference_type"],
            "P1→P2 (m)": r["width_m"],
            "P2→P3 (m)": r["length_m"],
            "Qualità geometrica": r["quality"],
            "Larghezza: origine": ORIGINS.get(r["width_origin"]),
            "Lunghezza: origine": ORIGINS.get(r["length_origin"]),
            "Scala avanzata": ORIGINS.get(r["scale_origin"]),
            "Revisione": r["revision"],
            "Preparazione": PREPARATION[r["preparation"]],
            "Motivo": r["message"],
        }
        for r in rows
    ]
    event = st.dataframe(
        display,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="prep_table_" + job.name,
    )
    selected_rows = list(event.selection.rows)
    readiness = analysis_readiness(job, min_quality=cfg.calibration.min_confidence)
    if readiness["finished"]:
        st.write(
            f"**Riepilogo:** {len(rows) - len(readiness['failed'])}/{len(rows)} video utilizzabili."
        )
        if readiness["failed"]:
            st.warning(
                "La preparazione ha esaminato tutti i video. Puoi correggere a mano i casi falliti oppure usare solo quelli calibrati automaticamente."
            )
            st.dataframe(
                [
                    {
                        "Video da calibrare a mano": r["video"],
                        "Motivo": r["message"] or "Qualità insufficiente",
                    }
                    for r in readiness["failed"]
                ],
                hide_index=True,
            )
        else:
            st.success("Tutte le calibrazioni sono utilizzabili. L'analisi non è ancora partita.")

        def open_analysis():
            st.session_state["metric_batch_stage"] = "Analisi batch"

        automatic_count = sum(
            r["state"] == "confirmed"
            and r["acceptance"] == "automatic"
            and (r["quality"] or 0) >= readiness["min_quality"]
            for r in rows
        )

        def select_automatic():
            try:
                with st.spinner("Salvataggio del campione comune e delle calibrazioni..."):
                    create_automatic_selection(job)
                st.session_state.pop("prep_selection_error", None)
                open_analysis()
            except (OSError, ValueError, RuntimeError) as exc:
                st.session_state["prep_selection_error"] = str(exc)

        st.button(
            f"Usa solo i {automatic_count} video calibrati automaticamente",
            key="prep_use_automatic",
            on_click=select_automatic,
            disabled=automatic_count == 0,
            help="Salva lo stesso elenco per il batch con omografia e per quello di sola segmentazione. L'analisi si avvia dal pulsante dedicato.",
        )
        if st.session_state.get("prep_selection_error"):
            st.error(st.session_state["prep_selection_error"])
        st.button(
            "Passa all'avvio dell'analisi",
            key="prep_analysis",
            on_click=open_analysis,
            disabled=not readiness["ready"],
        )
    event_key = "prep_table_selection_" + job.name
    if selected_rows != st.session_state.get(event_key, []):
        st.session_state[event_key] = selected_rows
        if selected_rows:
            st.session_state[select_key] = selected_rows[0]
    selected = st.selectbox(
        "Video da revisionare",
        range(len(rows)),
        format_func=lambda i: rows[i]["video"],
        key=select_key,
    )
    if not cursor_path.is_file() or read_json(cursor_path).get("index") != selected:
        write_json(cursor_path, {"index": selected})

    def next_video():
        following = list(range(selected + 1, len(rows))) + list(range(selected))
        st.session_state[select_key] = next(
            (i for i in following if rows[i]["state"] != "confirmed"), selected
        )

    st.button(
        "Prossimo video da revisionare",
        on_click=next_video,
        key="prep_next",
        disabled=not any(r["state"] != "confirmed" and r["index"] != selected for r in rows),
    )
    if snapshot(job)["active"]:
        st.info("Attendi il termine o ferma le proposte prima di modificare le calibrazioni.")
        return
    namespace = f"prep_{job.name}_{selected}"
    if refresh:
        st.session_state.pop(namespace + "_calibration_editor", None)
    try:
        proposal = load_proposal(job, selected)
        render_calibration_editor(
            Path(manifest["folder"]) / rows[selected]["video"],
            root,
            namespace=namespace,
            min_confidence=readiness["min_quality"],
            prepared_proposal=(manifest["session_id"], proposal) if proposal else None,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Editor non disponibile: {exc}")
