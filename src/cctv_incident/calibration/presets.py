"""Opt-in US marking hypotheses, never measurements inferred from image pixels."""

import cv2
import numpy as np

from .coordinates import validate_quad
from .records import Distance, RecordModel

SOURCE_URL = "https://mutcd.fhwa.dot.gov/pdfs/11th_Editionr1/mutcd11theditionr1hl.pdf"
SOURCE_EDITION = "FHWA MUTCD 11th Edition with Revision 1, December 2025"


class MarkingPreset(RecordModel):
    id: str
    label: str
    marking_kind: str
    section: str
    width_range_m: tuple[float, float]
    length_m: float | None = None
    condition: str


PRESETS = (
    MarkingPreset(
        id="us_broken_10ft",
        label="USA · tratto discontinuo ordinario: 10 ft",
        marking_kind="broken_line",
        section="3A.04",
        width_range_m=(0.1016, 0.1524),
        length_m=3.048,
        condition="Un solo tratto completo di linea discontinua ordinaria, senza intervalli né altri tratti. Sono ammesse proporzioni diverse dalla guida 10/30 ft.",
    ),
    MarkingPreset(
        id="us_dotted_lane_3ft",
        label="USA · tratto di linea punteggiata di corsia: 3 ft",
        marking_kind="dotted_lane",
        section="3A.04",
        width_range_m=(0.1016, 0.1524),
        length_m=0.9144,
        condition="Un solo tratto completo di linea punteggiata di corsia di larghezza normale; non una linea larga né un'estensione dentro un'intersezione.",
    ),
    MarkingPreset(
        id="us_dotted_extension_2ft",
        label="USA · estensione punteggiata in intersezione: 2 ft",
        marking_kind="dotted_extension",
        section="3A.04",
        width_range_m=(0.1016, 0.1524),
        length_m=0.6096,
        condition="Un solo tratto completo di estensione di una linea normale nell'intersezione, raccordo o rampa; non una linea di corsia punteggiata.",
    ),
    MarkingPreset(
        id="us_crosswalk_bar",
        label="USA · barra longitudinale di attraversamento: larghezza 12–24 in",
        marking_kind="crosswalk_bar",
        section="3C.06",
        width_range_m=(0.3048, 0.6096),
        condition="Una sola barra bianca longitudinale completa di un attraversamento. La sua lunghezza non è determinata dal minimo normativo dell'attraversamento.",
    ),
)


def preset_changes(
    record, preset_id, *, usa_verified, marking_kind, reference_verified, assumed_width=None
):
    preset = next((p for p in PRESETS if p.id == preset_id), None)
    if preset is None:
        raise ValueError("Preset sconosciuto")
    if not usa_verified:
        raise ValueError("Verifica esplicitamente la provenienza USA; World e UAE non la attestano")
    if marking_kind != preset.marking_kind or not reference_verified:
        raise ValueError(
            "Verifica tipo di segnaletica, riferimento completo e corrispondenza dei lati"
        )
    candidate = record.automation.diagnostics.get("selected_candidate", {})
    if (
        record.geometry_mode != "rectangle"
        or candidate.get("reference_type") != "painted_bar"
        or not candidate.get("diagnostics", {}).get("single_marking")
    ):
        raise ValueError(
            "Il preset richiede una singola barra dipinta completa; non un rettangolo fra più tratti"
        )
    current = validate_quad(record.points_px, record.video.image_size).astype(np.float32)
    initial = validate_quad(candidate["points_px"], record.video.image_size).astype(np.float32)
    overlap, _ = cv2.intersectConvexConvex(current, initial)
    if (
        overlap / abs(cv2.contourArea(current)) < 0.8
        or overlap / abs(cv2.contourArea(initial)) < 0.5
    ):
        raise ValueError(
            "I punti non delimitano più il singolo riferimento proposto; specifica le misure manualmente"
        )
    if assumed_width is not None and (
        not np.isfinite(assumed_width)
        or not preset.width_range_m[0] <= assumed_width <= preset.width_range_m[1]
    ):
        raise ValueError("La larghezza ipotizzata deve essere nell'intervallo del preset")
    source = f"{SOURCE_EDITION}, §{preset.section}; {SOURCE_URL}. Ipotesi da verificare sulle specifiche locali e alla data della ripresa. {preset.condition}"
    changes = {}
    for name in ("width", "length"):
        old = getattr(record, name)
        if old.origin == "standard" and old.preset and old.preset != preset.id:
            changes[name] = Distance().model_dump()
    if assumed_width is not None:
        changes["width"] = Distance(
            value=assumed_width,
            origin="standard",
            source=source,
            preset=preset.id,
            user_confirmed=False,
        ).model_dump()
    if preset.length_m is not None:
        changes["length"] = Distance(
            value=preset.length_m,
            origin="standard",
            source=source,
            preset=preset.id,
            user_confirmed=False,
        ).model_dump()
    if preset.length_m is None and assumed_width is None:
        raise ValueError(
            "Seleziona una larghezza nell'intervallo; la lunghezza della barra resta da misurare"
        )
    automation = record.automation.model_dump()
    automation["diagnostics"]["preset_application"] = {
        "preset": preset.model_dump(),
        "source_url": SOURCE_URL,
        "source_edition": SOURCE_EDITION,
        "source_sha256": "f508594285e5fccc45714660cc6aba1f895ae88226449345582fbbc1ceb7ba07",
        "usa_verified": True,
        "marking_kind": marking_kind,
        "reference_verified": True,
        "assumed_width_m": assumed_width,
        "points_px_at_application": record.points_px,
    }
    changes["automation"] = automation
    return changes
