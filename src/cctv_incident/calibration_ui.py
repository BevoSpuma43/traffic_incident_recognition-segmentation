"""Reusable first-frame calibration editor for a single local video."""

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import streamlit as st
from pydantic import ValidationError

from cctv_incident.calibration.editor import EditorSession, geometric_quality, metric_preview
from cctv_incident.calibration.records import Distance, ExplicitScale, confirm_record, edit_record
from cctv_incident.calibration.repository import (
    compatibility_reasons,
    create_draft,
    find_record,
    inspect_video,
    load_record,
    save_record,
)
from cctv_incident.calibration_proposal_ui import render_automatic_proposals
from cctv_incident.components.calibration_editor import calibration_editor

ORIGINS = {
    "unknown": "Da specificare",
    "measured": "Misurata sul posto",
    "standard": "Valore da una fonte / standard",
    "experimental": "Ipotesi sperimentale",
}


def _validation_message(exc):
    if isinstance(exc, cv2.error):
        return "La geometria non consente di calcolare un'omografia stabile. Controlla i punti."
    message = (
        exc.errors(include_url=False)[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
    )
    message = message.removeprefix("Value error, ")
    messages = {
        "At least four x/y correspondences are required": "Seleziona quattro vertici del rettangolo.",
        "Expected four vertices in P1, P2, P3, P4 order": "Completa P1, P2, P3 e P4 in ordine lungo il perimetro.",
        "Vertices must be convex and ordered without crossings or flat corners": "I punti devono formare un quadrilatero convesso, senza lati incrociati o vertici allineati.",
        "Both metric distances require values and confirmed provenance": "Inserisci larghezza e lunghezza, specifica l'origine e conferma entrambe le misure.",
        "Explicit metric coordinates require confirmed scale provenance": "Conferma la scala e l'origine delle coordinate metriche.",
    }
    return messages.get(message, message)


@dataclass(frozen=True)
class EditorSelection:
    ready: bool = False
    path: Path | None = None
    camera_id: str | None = None
    record_id: str | None = None
    revision: int | None = None


@st.cache_data(show_spinner=False, max_entries=8)
def _display_image(frame):
    height, width = frame.shape[:2]
    if width > 1280:
        frame = cv2.resize(frame, (1280, round(height * 1280 / width)))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Impossibile visualizzare il fotogramma")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode()


def _verify_source(session, source, project_root):
    reasons = compatibility_reasons(session.record, inspect_video(source, project_root))
    if reasons:
        session.visually_accepted = False
        raise ValueError("Il video è cambiato: selezionalo nuovamente prima di confermare.")


def _distance(session, name, label, key):
    old = getattr(session.record, name)
    value = st.number_input(
        label,
        min_value=0.001,
        value=old.value,
        format="%.3f",
        key=key + "_value",
        disabled=session.locked,
    )
    origin = st.selectbox(
        "Origine della misura",
        list(ORIGINS),
        index=list(ORIGINS).index(old.origin),
        format_func=ORIGINS.get,
        key=key + "_origin",
        disabled=session.locked,
    )
    source = (
        st.text_input(
            "Fonte o descrizione della misura",
            old.source or "",
            key=key + "_source",
            disabled=session.locked,
        ).strip()
        or None
    )
    same = (value, origin, source) == (old.value, old.origin, old.source)
    # A distinct checkbox identity forces a fresh acknowledgement after a measure changes.
    token = json.dumps([value, origin, source], ensure_ascii=True)
    confirmed = st.checkbox(
        "Confermo valore e origine",
        value=old.user_confirmed if same else False,
        key=key + "_confirmed_" + token,
        disabled=session.locked,
    )
    try:
        return Distance(
            value=value,
            origin=origin,
            source=source,
            user_confirmed=confirmed,
            preset=old.preset if (origin, source) == (old.origin, old.source) else None,
        )
    except ValueError:
        st.warning("Indica un valore, la sua origine e, per valori assunti, una fonte.")
        return None


def _numeric_points(session, prefix):
    record = session.record
    with st.expander("Coordinate numeriche e corrispondenze avanzate"):
        if record.geometry_mode == "rectangle":
            st.caption("Alternativa ai clic: applica le coordinate prima di confermare.")
            with st.form(prefix + "_numeric"):
                existing = {v.id: v for v in record.vertices}
                points = []
                for i in range(1, 5):
                    vertex = existing.get(f"P{i}")
                    cols = st.columns(2)
                    x = cols[0].number_input(
                        f"P{i} · X",
                        min_value=0.0,
                        max_value=float(record.video.image_size[0] - 1),
                        value=vertex.x if vertex else None,
                        key=prefix + f"_x{i}",
                        disabled=session.locked,
                    )
                    y = cols[1].number_input(
                        f"P{i} · Y",
                        min_value=0.0,
                        max_value=float(record.video.image_size[1] - 1),
                        value=vertex.y if vertex else None,
                        key=prefix + f"_y{i}",
                        disabled=session.locked,
                    )
                    if x is not None and y is not None:
                        points.append({"id": f"P{i}", "x": x, "y": y})
                apply = st.form_submit_button("Applica coordinate", disabled=session.locked)
            if apply:
                session.change(vertices=points)
                st.rerun()
        with st.form(prefix + "_advanced"):
            st.caption(
                "Corrispondenze esplicite: stesso ordine in entrambe le liste; coordinate sul piano in metri."
            )
            src = st.text_area(
                "Punti immagine [x, y] (JSON)",
                json.dumps(record.points_px),
                key=prefix + "_src",
                disabled=session.locked,
            )
            dst = st.text_area(
                "Punti sul piano [x, y] in metri (JSON)",
                json.dumps(record.destination_points),
                key=prefix + "_dst",
                disabled=session.locked,
            )
            scale = record.explicit_scale
            origin = st.selectbox(
                "Origine della scala metrica",
                ["measured", "standard", "experimental"],
                index=["measured", "standard", "experimental"].index(scale.origin) if scale else 0,
                format_func=ORIGINS.get,
                key=prefix + "_scale_origin",
                disabled=session.locked,
            )
            source = st.text_input(
                "Fonte della scala metrica",
                scale.source if scale else "",
                key=prefix + "_scale_source",
                disabled=session.locked,
            )
            # Explicit application always starts with an unconfirmed scale.
            apply = st.form_submit_button(
                "Applica corrispondenze avanzate", disabled=session.locked
            )
        if apply:
            try:
                vertices = [
                    {"id": f"P{i}", "x": x, "y": y} for i, (x, y) in enumerate(json.loads(src), 1)
                ]
                session.change(
                    geometry_mode="explicit",
                    vertices=vertices,
                    destination_points=json.loads(dst),
                    explicit_scale=ExplicitScale(origin=origin, source=source.strip()).model_dump(),
                )
                st.rerun()
            except (ValueError, TypeError) as exc:
                session.visually_accepted = False
                st.error(f"Corrispondenze non valide: {exc}")


def render_calibration_editor(
    source, project_root, *, namespace="single", min_confidence=0.55, prepared_proposal=None
):
    """Return a saved, explicitly reviewed selection; never borrow another video's record."""
    project_root = Path(project_root)
    state_key = namespace + "_calibration_editor"
    root = project_root / "data/calibration/videos"
    empty = EditorSelection()
    try:
        path = Path(source).resolve()
        stat = path.stat()
        fingerprint = (str(path), stat.st_size, stat.st_mtime_ns, str(project_root.resolve()))
        session = st.session_state.get(state_key)
        if session is None or session.fingerprint != fingerprint:
            st.session_state.pop(state_key, None)
            with st.spinner("Lettura del primo fotogramma e verifica del video..."):
                record, frame = create_draft(path, project_root)
                match = find_record(record.video, root)
                session = EditorSession(fingerprint, record, frame)
                if match.status == "compatible":
                    session.record, session.path = match.record, match.path
                    session.locked = match.record.status == "confirmed"
                elif match.status in {"incompatible", "ambiguous"}:
                    session.input_error = "La calibrazione archiviata non è associabile in modo univoco a questo video. Crea una nuova bozza."
                if prepared_proposal is not None:
                    preparation_id, result = prepared_proposal
                    if (
                        session.record.automation.diagnostics.get("preparation_session")
                        == preparation_id
                        and result.diagnostics["reference_pixel_sha256"]
                        == session.record.reference.pixel_sha256
                        and session.record.roi_px is None
                    ):
                        session.proposal_result = result
                if session.record.automation.method == "auto_assisted":
                    session.workflow = "automatic"
                    selected = session.record.automation.diagnostics.get("selected_candidate", {})
                    if selected.get("candidate_id"):
                        session.applied_proposal = (
                            session.proposal_generation,
                            selected["candidate_id"],
                        )
                st.session_state[state_key] = session
    except (OSError, ValueError) as exc:
        st.session_state.pop(state_key, None)
        st.info(f"Seleziona un video locale leggibile per calibrarlo. {exc}")
        return empty

    # Streamlit can retain a session created by the previous editor class on hot reload.
    if not hasattr(session, "workflow"):
        session.workflow = (
            "automatic" if session.record.automation.method == "auto_assisted" else "manual"
        )
        selected = session.record.automation.diagnostics.get("selected_candidate", {})
        session.applied_proposal = (
            (session.proposal_generation, selected["candidate_id"])
            if selected.get("candidate_id")
            else None
        )
        session.proposal_error = None
    record = session.record
    prefix = f"{namespace}_{record.record_id}_{session.epoch}"
    st.caption(
        f"{path.name} · {record.video.image_size[0]} × {record.video.image_size[1]} pixel originali "
        f"· revisione {record.revision} · {record.status}"
    )
    if session.input_error:
        st.warning(session.input_error)

    if session.locked:
        st.info("Controlla punti e misure sul fotogramma prima di riutilizzare la calibrazione.")
        if st.button("Modifica calibrazione", key=prefix + "_unlock"):
            session.record = edit_record(record)
            session.locked = False
            session.visually_accepted = False
            session.epoch += 1
            st.rerun()
        if st.button("Riutilizza calibrazione", key=prefix + "_reuse"):
            try:
                _verify_source(session, path, project_root)
                # Verify the saved image and current revision again, too.
                saved = load_record(session.path)
                if saved != record:
                    raise ValueError("La calibrazione salvata è cambiata: ricarica il video.")
                session.visually_accepted = True
            except (OSError, ValueError) as exc:
                session.visually_accepted = False
                st.error(str(exc))

    render_automatic_proposals(session, prefix, namespace)

    automatic = session.workflow == "automatic"
    if automatic and (
        session.proposal_error
        or (session.proposal_result is not None and not session.proposal_result.candidates)
    ):
        return empty
    if automatic:
        st.write(
            "Proposta automatica: trascina i punti per correggerli e modifica le distanze sotto il fotogramma."
        )
        st.caption(
            "P1→P2 è la larghezza; P2→P3 è la lunghezza. Le distanze disponibili sono precompilate; quelle sconosciute restano da inserire."
        )
        mode = "calibration"
    else:
        st.write(
            "Seleziona P1–P4 lungo il perimetro di un rettangolo sul piano stradale. "
            "P1→P2 è la larghezza; P2→P3 è la lunghezza. La ROI è indipendente."
        )
        mode = st.radio(
            "Modifica sul fotogramma",
            ["calibration", "roi"],
            horizontal=True,
            format_func=lambda m: "Punti di calibrazione" if m == "calibration" else "ROI stradale",
            key=namespace + "_mode_" + record.record_id,
            disabled=session.locked,
        )
    component_key = f"{namespace}_canvas_{record.video.sha256}_{record.record_id}_{record.revision}"

    def receive_edit():
        try:
            result = st.session_state.get(component_key)
            session.receive(result.get("edit") if result else None)
        except (ValueError, TypeError, KeyError) as exc:
            session.visually_accepted = False
            session.input_error = f"Modifica non valida: {exc}"

    calibration_editor(
        data={
            "record_id": record.record_id,
            "source_sha256": record.video.sha256,
            "revision": record.revision,
            "epoch": session.epoch,
            "image": _display_image(session.frame),
            "image_size": record.video.image_size,
            "vertices": [
                v.model_dump() for v in sorted(record.vertices, key=lambda v: int(v.id[1:]))
            ],
            "roi_px": record.roi_px,
            "width_m": record.width.value,
            "length_m": record.length.value,
            "geometry_mode": record.geometry_mode,
            "mode": mode,
            "readonly": session.locked,
            "max_points": 4 if record.geometry_mode == "rectangle" else len(record.vertices),
        },
        key=component_key,
        on_edit=receive_edit,
    )

    cols = st.columns(1 if automatic else 4)

    def new_roi():
        session.change(roi_px=[])
        st.session_state[namespace + "_mode_" + record.record_id] = "roi"

    if cols[0].button(
        "Annulla ultima modifica",
        disabled=session.locked or not session.history,
        key=prefix + "_undo",
    ):
        session.undo()
        st.rerun()
    if not automatic and cols[1].button(
        "Azzera punti", disabled=session.locked, key=prefix + "_reset"
    ):
        session.change(
            vertices=[],
            geometry_mode="rectangle",
            destination_points=[],
            explicit_scale=None,
            automation={},
        )
        st.rerun()
    if not automatic:
        cols[2].button("Nuova ROI", disabled=session.locked, key=prefix + "_roi", on_click=new_roi)
    if not automatic and cols[3].button(
        "ROI: tutto il frame", disabled=session.locked, key=prefix + "_full"
    ):
        session.change(roi_px=None)
        st.rerun()

    if not automatic:
        _numeric_points(session, prefix)
    inputs_valid = True
    if record.geometry_mode == "rectangle":
        cols = st.columns(2)
        with cols[0]:
            width = _distance(session, "width", "Larghezza P1→P2 (m)", prefix + "_width")
        with cols[1]:
            length = _distance(session, "length", "Lunghezza P2→P3 (m)", prefix + "_length")
        inputs_valid = width is not None and length is not None
        if not session.locked:
            if inputs_valid:
                if session.change(width=width.model_dump(), length=length.model_dump()):
                    st.rerun()
            else:
                session.visually_accepted = False
    elif record.explicit_scale:
        checked = st.checkbox(
            "Confermo la scala e la provenienza delle coordinate metriche",
            value=record.explicit_scale.user_confirmed,
            key=prefix + "_scale_confirm",
            disabled=session.locked,
        )
        if not session.locked and checked != record.explicit_scale.user_confirmed:
            scale = record.explicit_scale.model_copy(update={"user_confirmed": checked})
            session.change(explicit_scale=scale.model_dump())
            st.rerun()

    confirmed = None
    try:
        preview, step = metric_preview(session.frame, record)
        st.image(preview, channels="BGR", caption=f"Anteprima metrica · griglia ogni {step:g} m")
        quality = geometric_quality(record)
        st.caption(
            f"Qualità geometrica indicativa: {quality:.2f}. Non misura l'accuratezza fisica delle distanze."
        )
        candidate = record if session.locked else edit_record(record, geometric_quality=quality)
        confirmed = confirm_record(candidate)
    except (ValueError, cv2.error) as exc:
        st.info(f"Calibrazione da completare o correggere: {_validation_message(exc)}")

    reviewed = st.checkbox(
        "Ho verificato il fotogramma, le corrispondenze e le misure sul piano stradale",
        key=prefix + "_review",
        disabled=session.locked,
    )
    cols = st.columns(2)
    if cols[0].button(
        "Salva bozza", disabled=session.locked or not inputs_valid, key=prefix + "_draft"
    ):
        try:
            _verify_source(session, path, project_root)
            session.path = save_record(record, root, reference_image=session.frame)
            session.record = load_record(session.path)
            session.visually_accepted = False
            session.epoch += 1
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(f"Bozza non salvata: {exc}")
    if cols[1].button(
        "Conferma e salva calibrazione",
        type="primary",
        disabled=session.locked or not inputs_valid or confirmed is None or not reviewed,
        key=prefix + "_confirm",
    ):
        try:
            _verify_source(session, path, project_root)
            session.path = save_record(confirmed, root, reference_image=session.frame)
            session.record = load_record(session.path)
            session.locked = True
            session.visually_accepted = True
            session.epoch += 1
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(f"Calibrazione non salvata: {exc}")

    ready = (
        session.path is not None
        and session.visually_accepted
        and session.record.status == "confirmed"
        and session.record.runtime.metric_valid(min_confidence)
    )
    if session.record.status == "confirmed":
        if ready:
            st.success("Calibrazione confermata per questo video: analisi metrica disponibile.")
        elif session.visually_accepted:
            st.warning(
                f"Qualità geometrica inferiore alla soglia richiesta ({min_confidence:.2f}): "
                "controlla i punti e la soglia nella configurazione."
            )
    return EditorSelection(
        ready,
        session.path if ready else None,
        session.record.camera_id if ready else None,
        session.record.record_id if ready else None,
        session.record.revision if ready else None,
    )
