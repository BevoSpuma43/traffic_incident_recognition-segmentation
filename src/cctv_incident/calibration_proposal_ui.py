"""Proposal selection and opt-in dimension hypotheses in the shared editor."""

import cv2
import numpy as np
import streamlit as st

from .calibration.presets import PRESETS, SOURCE_EDITION, SOURCE_URL, preset_changes
from .calibration.proposals import candidate_changes, generate_proposals
from .calibration.temporal import search_initial_frames
from .calibration.vehicle_geometry import VehicleDimensions
from .calibration.vehicle_search import generate_vehicle_proposals

REFERENCE_LABELS = {
    "us_crosswalk": "Attraversamento · ipotesi USA",
    "us_dash_group": "Tratti consecutivi · ipotesi USA",
    "us_single_dash": "Singolo tratto · ipotesi USA ambigua",
    "us_lane_and_dash": "Corsia e tratto · ipotesi USA",
    "painted_bar": "Barra dipinta",
    "outlined_rectangle": "Rettangolo delimitato",
    "reconstructed_bar": "Barra ricostruita",
    "repeated_bars": "Gruppo di barre affiancate",
    "repeated_dashes": "Gruppo di tratteggi allineati",
    "vehicle_reference": "Piano stradale da automobili · sperimentale",
}

MARKING_KINDS = {
    "unknown": "Da verificare",
    "broken_line": "Tratto discontinuo ordinario",
    "dotted_lane": "Tratto punteggiato di corsia (linea normale)",
    "dotted_extension": "Estensione punteggiata nell'intersezione",
    "crosswalk_bar": "Barra longitudinale di attraversamento",
}


def render_automatic_proposals(session, prefix, namespace):
    st.caption(
        "L'automatismo cerca barre, tratteggi e gruppi di segnaletica sul primo fotogramma. "
        "Controlla che il riferimento sia sul piano stradale: oggetti e superfici rialzate possono confonderlo."
    )
    if st.button("Calibrazione automatica", key=prefix + "_automatic", disabled=session.locked):
        session.workflow = "automatic"
        session.proposal_result = None
        session.proposal_error = None
        session.applied_proposal = None
        try:
            with st.spinner("Ricerca dei riferimenti stradali..."):
                session.proposal_result = generate_proposals(session.frame, session.record.roi_px)
                session.proposal_generation += 1
        except (ValueError, cv2.error):
            session.proposal_error = "La ROI o il fotogramma non consentono di cercare una proposta. Completa la ROI o usa l'editor manuale."
        st.rerun()
    with st.expander("Ricerca su più fotogrammi"):
        st.caption(
            "Cerca riferimenti scoperti dai veicoli nei primi secondi. "
            "Scegli un intervallo prima dell'incidente. I fotogrammi con camera mossa "
            "o sfondo non verificabile vengono scartati."
        )
        seconds = st.number_input(
            "Secondi iniziali da esaminare",
            min_value=0.5,
            max_value=15.0,
            value=5.0,
            step=0.5,
            key=f"{namespace}_{session.record.record_id}_search_seconds",
            disabled=session.locked,
        )
        if st.button("Cerca nei primi secondi", key=prefix + "_temporal", disabled=session.locked):
            session.workflow = "automatic"
            session.proposal_result = None
            session.proposal_error = None
            session.applied_proposal = None
            try:
                with st.spinner("Ricerca della segnaletica nei fotogrammi iniziali..."):
                    session.proposal_result = search_initial_frames(
                        session.fingerprint[0],
                        session.frame,
                        session.record.roi_px,
                        seconds=seconds,
                    )
                    session.proposal_generation += 1
            except (OSError, ValueError, cv2.error) as exc:
                session.proposal_error = f"Ricerca su più fotogrammi non disponibile: {exc}"
            st.rerun()
    _vehicle_controls(session, prefix, namespace)
    if session.workflow == "automatic":
        if st.button(
            "Torna alla selezione manuale", key=prefix + "_manual", disabled=session.locked
        ):
            session.workflow = "manual"
            st.rerun()
        if session.proposal_error:
            st.warning(session.proposal_error)
    if session.workflow != "automatic":
        return
    result = session.proposal_result
    if result is None:
        selected = session.record.automation.diagnostics.get("selected_candidate", {})
        if not session.proposal_error and selected.get("reference_type") == "painted_bar":
            _preset_controls(session, prefix, selected["candidate_id"])
        return
    if not result.candidates:
        st.info(result.diagnostics["reason"])
    else:
        key = f"{namespace}_{session.record.record_id}_proposal_{session.proposal_generation}"

        def label(id):
            return next(
                f"{i}. {REFERENCE_LABELS.get(c.reference_type, c.reference_type)} · qualità {c.geometric_quality:.2f}"
                for i, c in enumerate(result.candidates, 1)
                if c.candidate_id == id
            )

        candidate_id = st.selectbox(
            "Proposta da esaminare",
            [c.candidate_id for c in result.candidates],
            index=next(
                (
                    i
                    for i, c in enumerate(result.candidates)
                    if c.candidate_id
                    == session.record.automation.diagnostics.get("selected_candidate", {}).get(
                        "candidate_id"
                    )
                ),
                0,
            ),
            key=key + "_choice",
            disabled=session.locked,
            format_func=label,
        )
        candidate = next(c for c in result.candidates if c.candidate_id == candidate_id)
        st.write(" ".join(candidate.reasons))
        source_frame = candidate.diagnostics.get("source_frame")
        if source_frame:
            st.caption(
                f"Riferimento trovato a {source_frame['timestamp_s']:.2f} s; punti sul primo fotogramma."
            )
            evidence = result.evidence_images.get(source_frame["pixel_sha256"])
            if evidence is not None:
                evidence = evidence.copy()
                ratios = np.array(session.record.video.image_size) / np.array(
                    [evidence.shape[1], evidence.shape[0]]
                )
                quad = np.rint(np.asarray(candidate.points_px) / ratios).astype(np.int32)
                cv2.polylines(evidence, [quad], True, (0, 255, 0), 2)
                st.image(
                    evidence,
                    channels="BGR",
                    caption="Fotogramma in cui il riferimento è visibile: controlla i quattro punti.",
                )
        if candidate.reference_type == "vehicle_reference":
            st.caption(
                "Scala approssimativa basata sull'auto tipo. Verifica il rettangolo stradale e conferma l'origine sperimentale delle distanze."
            )
            for frame_index, evidence in result.evidence_images.items():
                if frame_index != "0":
                    st.image(
                        evidence,
                        channels="BGR",
                        caption=f"Osservazioni delle automobili · fotogramma {frame_index} · modello 3D in azzurro.",
                    )
        token = (session.proposal_generation, candidate_id)
        if not session.locked and session.applied_proposal != token:
            if (
                result.diagnostics["reference_pixel_sha256"]
                != session.record.reference.pixel_sha256
            ):
                session.proposal_result = None
                session.proposal_error = (
                    "La proposta appartiene a un altro fotogramma: rigenerala per questo video."
                )
                st.rerun()
            else:
                session.change(**candidate_changes(candidate, result))
                session.applied_proposal = token
                st.rerun()
        selected = session.record.automation.diagnostics.get("selected_candidate", {})
        if selected.get("reference_type") == "painted_bar":
            _preset_controls(session, prefix, selected["candidate_id"])
    with st.expander("Diagnostica della proposta automatica"):
        discarded = result.diagnostics.get("discarded", {})
        if discarded:
            st.caption(
                "Riferimenti scartati durante la ricerca: "
                + "; ".join(
                    f"{reason.replace('_', ' ')}: {count}" for reason, count in discarded.items()
                )
            )
        temporal = result.diagnostics.get("temporal_search")
        if temporal:
            accepted = sum(row["accepted"] for row in temporal["frames"])
            st.caption(
                f"Fotogrammi aggiuntivi verificati: {accepted} su {len(temporal['frames'])} esaminati."
            )
            st.caption(
                f"Osservazioni scartate perché prive di conferma temporale: {temporal.get('unconfirmed_observations', 0)}."
            )
        st.image(
            result.preview,
            channels="BGR",
            caption="Linee rilevate e candidati; verificare il riferimento sul fotogramma nell'editor.",
        )
        st.image(
            result.mask,
            caption="Maschera delle automobili candidate"
            if result.diagnostics.get("scale_origin") == "experimental"
            else "Maschera della segnaletica candidata",
        )
        if not result.candidates and result.diagnostics.get("scale_origin") == "experimental":
            for frame_index, evidence in result.evidence_images.items():
                if frame_index != "0":
                    st.image(
                        evidence,
                        channels="BGR",
                        caption=f"Osservazioni valutate · fotogramma {frame_index} · sagome in arancione.",
                    )
        st.json(result.diagnostics, expanded=False)


def _vehicle_model_paths(project_root):
    from pathlib import Path

    folders = [Path(project_root) / "models", Path.cwd() / "models"]
    return sorted(
        {p.resolve() for folder in folders for p in folder.glob("*seg.pt") if p.is_file()},
        key=lambda p: ("n-seg" not in p.name, p.name),
    )


def _vehicle_controls(session, prefix, namespace):
    key = f"{namespace}_{session.record.record_id}_vehicles"
    parameters = session.record.automation.diagnostics.get("parameters", {})
    selected = session.record.automation.diagnostics.get("selected_candidate", {})
    dimensions = VehicleDimensions()
    if selected.get("reference_type") == "vehicle_reference":
        dimensions = VehicleDimensions.model_validate(parameters.get("dimensions", {}))
    with st.expander("Calibrazione dalle automobili · sperimentale"):
        st.caption(
            "Ipotesi per il progetto: auto tipo 4,7 × 1,8 × 1,5 m. "
            "Non è una media statistica verificata del parco auto USA. "
            "Il modello assume strada planare e rettilinea, camera fissa senza rollio "
            "e automobili con direzioni parallele."
        )
        cols = st.columns(3)
        values = []
        for col, label, initial, lower, upper, suffix in zip(
            cols,
            ["Lunghezza auto tipo (m)", "Larghezza auto tipo (m)", "Altezza auto tipo (m)"],
            [dimensions.length_m, dimensions.width_m, dimensions.height_m],
            [3.0, 1.4, 1.1],
            [6.5, 2.5, 2.5],
            ["length", "width", "height"],
            strict=True,
        ):
            values.append(
                col.number_input(
                    label,
                    min_value=lower,
                    max_value=upper,
                    value=initial,
                    step=0.1,
                    key=key + suffix,
                    disabled=session.locked,
                )
            )
        seconds = st.number_input(
            "Secondi iniziali per osservare le auto",
            min_value=0.0,
            max_value=15.0,
            value=float(parameters.get("seconds", 5.0))
            if selected.get("reference_type") == "vehicle_reference"
            else 5.0,
            step=0.5,
            key=key + "_seconds",
            disabled=session.locked,
        )
        st.caption(
            "0 usa solo il primo fotogramma. Scegli un intervallo prima dell'incidente; modificare i valori richiede una nuova ricerca."
        )
        models = _vehicle_model_paths(session.fingerprint[3])
        model = st.selectbox(
            "Modello locale per riconoscere le automobili",
            models or [None],
            format_func=lambda path: path.name if path else "Nessun modello locale disponibile",
            key=key + "_model",
            disabled=session.locked,
        )
        if st.button(
            "Usa automobili (sperimentale)",
            key=prefix + "_vehicles_auto",
            disabled=session.locked or model is None,
        ):
            session.workflow = "automatic"
            session.proposal_result = None
            session.proposal_error = None
            session.applied_proposal = None
            try:
                dimensions = VehicleDimensions(
                    length_m=values[0], width_m=values[1], height_m=values[2]
                )
                with st.spinner(
                    "Riconoscimento delle automobili e stima sperimentale del piano stradale..."
                ):
                    session.proposal_result = generate_vehicle_proposals(
                        session.fingerprint[0],
                        session.frame,
                        session.record.roi_px,
                        model_path=model,
                        dimensions=dimensions,
                        seconds=seconds,
                    )
                    session.proposal_generation += 1
            except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
                session.proposal_error = f"Calibrazione dalle automobili non disponibile: {exc}"
            st.rerun()


def _preset_controls(session, prefix, candidate_id):
    with st.expander("Misure suggerite da preset USA"):
        st.caption(
            f"Ipotesi modificabili, non misure osservate. Riferimento nell'editor: {candidate_id}."
        )
        usa = st.checkbox(
            "Ho verificato la provenienza USA del video",
            key=prefix + "_usa",
            disabled=session.locked,
        )
        kind = st.selectbox(
            "Tipo di segnaletica verificato",
            list(MARKING_KINDS),
            format_func=MARKING_KINDS.get,
            key=prefix + "_marking_kind",
            disabled=session.locked,
        )
        preset_id = st.selectbox(
            "Preset dimensionale",
            [None] + [p.id for p in PRESETS],
            format_func=lambda id: (
                "Nessuno" if id is None else next(p.label for p in PRESETS if p.id == id)
            ),
            key=prefix + "_preset",
            disabled=session.locked,
        )
        if preset_id is None:
            return
        preset = next(p for p in PRESETS if p.id == preset_id)
        st.write(preset.condition)
        st.caption(
            f"{SOURCE_EDITION}, §{preset.section}. Verifica anche data della ripresa e specifiche locali."
        )
        st.link_button("Fonte FHWA del preset", SOURCE_URL)
        st.caption(
            f"Larghezza prevista: {preset.width_range_m[0]:g}–{preset.width_range_m[1]:g} m. "
            "L'intervallo non identifica una larghezza esatta."
        )
        width = st.number_input(
            "Larghezza ipotizzata del singolo tratto (m)",
            min_value=preset.width_range_m[0],
            max_value=preset.width_range_m[1],
            value=None,
            format="%.4f",
            key=prefix + "_assumed_width_" + preset.id,
            disabled=session.locked,
        )
        verified = st.checkbox(
            "Il rettangolo contiene un solo tratto completo: P1→P2 è la sua larghezza e P2→P3 la sua lunghezza",
            key=prefix + "_reference_verified",
            disabled=session.locked,
        )
        if st.button(
            "Applica preset come ipotesi",
            key=prefix + "_apply_preset",
            disabled=session.locked or not usa or not verified or kind != preset.marking_kind,
        ):
            try:
                session.change(
                    **preset_changes(
                        session.record,
                        preset_id,
                        usa_verified=usa,
                        marking_kind=kind,
                        reference_verified=verified,
                        assumed_width=width,
                    )
                )
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc))
