"""
src/collision_logic.py
======================
Algoritmo di rilevamento collisioni tra veicoli tracciati.

La logica combina due segnali cinematici con la verifica di sovrapposizione
delle maschere di segmentazione pixel-level:

  1. Overlap pixel-level → le maschere delle due istanze YOLO si sovrappongono
     per almeno ``mask_overlap_threshold`` pixel (indica prossimità fisica).

  2. Anomalia cinematica → almeno uno dei seguenti:
       a) Dual stop: entrambi i veicoli sono fermi da abbastanza frame consecutivi.
       b) Hard deceleration: almeno un veicolo decelera bruscamente.

Un evento di collisione viene generato solo quando si verificano sia (1)
che almeno una condizione di (2), riducendo i falsi positivi da occlusioni.

Dipendenze interne (bottom-up):
  src.config  → AppConfig (soglie di rilevamento)
  src.models  → CollisionEvent, VehicleState
  src.geometry → mask_intersection_area
"""

from __future__ import annotations

from src.config import AppConfig
from src.geometry import mask_intersection_area
from src.models import CollisionEvent, VehicleState


class CollisionDetector:
    """
    Rilevatore di possibili collisioni tra coppie di veicoli tracciati.

    Analizza ogni coppia unica di veicoli per ogni frame, verificando:
      - sovrapposizione delle maschere binarie pixel-level
      - anomalie cinematiche (stop prolungato o frenata brusca)

    Le soglie sono configurabili tramite AppConfig o parametri espliciti.
    """

    def __init__(
        self,
        mask_overlap_threshold: int,
        stopped_frames_threshold: int,
        stopped_speed_threshold: float,
        strong_deceleration_threshold: float,
    ) -> None:
        """
        Inizializza il rilevatore con le soglie di collisione.

        Parameters
        ----------
        mask_overlap_threshold : int
            Numero minimo di pixel sovrapposti tra due maschere per
            considerare la coppia in possibile collisione.
        stopped_frames_threshold : int
            Numero minimo di frame consecutivi in cui un veicolo deve
            risultare fermo affinché la condizione "dual stop" si attivi.
        stopped_speed_threshold : float
            Velocità massima (pixel/frame) al di sotto della quale un
            veicolo è considerato fermo.
        strong_deceleration_threshold : float
            Soglia di accelerazione (pixel/frame², valore negativo) al di
            sotto della quale si considera una frenata brusca anomala.
        """
        # Soglie salvate come attributi di istanza per essere accessibili
        # nei metodi privati di verifica delle condizioni.
        self.mask_overlap_threshold: int = mask_overlap_threshold
        self.stopped_frames_threshold: int = stopped_frames_threshold
        self.stopped_speed_threshold: float = stopped_speed_threshold
        self.strong_deceleration_threshold: float = strong_deceleration_threshold

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: AppConfig) -> CollisionDetector:
        """
        Crea un CollisionDetector a partire da un oggetto AppConfig.

        Factory method che elimina la dipendenza dall'ordine dei parametri
        posizionali quando si costruisce dall'oggetto di configurazione
        (come fa video_pipeline.py).

        Parameters
        ----------
        config : AppConfig
            Configurazione dell'applicazione con le soglie già valorizzate.

        Returns
        -------
        CollisionDetector
            Istanza con le soglie estratte dalla config.
        """
        return cls(
            mask_overlap_threshold=config.mask_overlap_threshold,
            stopped_frames_threshold=config.stopped_frames_threshold,
            stopped_speed_threshold=config.stopped_speed_threshold,
            strong_deceleration_threshold=config.strong_deceleration_threshold,
        )

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def detect_collisions(
        self,
        vehicle_states: list[VehicleState],
        frame_index: int,
    ) -> list[CollisionEvent]:
        """
        Rileva eventi di possibile collisione per il frame corrente.

        Itera su tutte le coppie uniche di veicoli (combinazioni senza
        ripetizione) e genera un CollisionEvent per ogni coppia che soddisfa
        sia la condizione di overlap sia quella cinematica.

        Parameters
        ----------
        vehicle_states : list[VehicleState]
            Lista degli stati attivi nel frame corrente, restituita da
            VehicleStateStore.update().
        frame_index : int
            Indice del frame corrente, salvato nell'evento per il logging.

        Returns
        -------
        list[CollisionEvent]
            Lista degli eventi di collisione rilevati. Può essere vuota
            se non ci sono sovrapposizioni con anomalie cinematiche.
        """
        collisions: list[CollisionEvent] = []

        # Set delle coppie già analizzate: evita di processare due volte
        # la stessa coppia nel caso (raro ma possibile) in cui il tracker
        # assegni lo stesso track_id a istanze diverse nella stessa lista.
        # La struttura del doppio loop (j > i) garantisce già unicità per
        # liste senza duplicati; il set agisce come guardia di sicurezza.
        seen_pairs: set[tuple[int, int]] = set()

        for i in range(len(vehicle_states)):
            vehicle_a = vehicle_states[i]

            for j in range(i + 1, len(vehicle_states)):
                vehicle_b = vehicle_states[j]

                # Crea una chiave di coppia ordinata (id_min, id_max) per
                # garantire l'idempotenza indipendentemente dall'ordine
                # in cui i veicoli appaiono nella lista.
                pair = self._pair_key(vehicle_a.track_id, vehicle_b.track_id)

                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # --- Condizione 1: sovrapposizione pixel-level delle maschere ---
                # Se le maschere non si sovrappongono abbastanza, la coppia
                # non è candidata a collisione: skip immediato.
                has_overlap, overlap_area = self._has_overlap(vehicle_a, vehicle_b)
                if not has_overlap:
                    continue

                # --- Condizione 2: anomalia cinematica ---
                # Almeno una delle due sub-condizioni deve essere vera.
                dual_stop = self._both_stopped(vehicle_a, vehicle_b)
                hard_deceleration = self._hard_deceleration(vehicle_a, vehicle_b)

                if not dual_stop and not hard_deceleration:
                    # Sovrapposizione senza anomalia cinematica: potrebbe
                    # essere un'occlusione normale o un artefatto del tracker.
                    continue

                # Costruisce la stringa descrittiva del motivo del rilevamento.
                # Usata per logging e per distinguere i casi nei report.
                if dual_stop and hard_deceleration:
                    reason = "overlap_and_dual_stop_and_hard_deceleration"
                elif dual_stop:
                    reason = "overlap_and_dual_stop"
                else:
                    reason = "overlap_and_hard_deceleration"

                collisions.append(
                    CollisionEvent(
                        frame_index=frame_index,
                        track_id_a=pair[0],
                        track_id_b=pair[1],
                        overlap_area=overlap_area,
                        reason=reason,
                        # speed_a, speed_b, acceleration_a, acceleration_b
                        # hanno default=0.0 in CollisionEvent: non li popoliamo
                        # qui per semplicità, sono disponibili per estensioni future.
                    )
                )

        return collisions

    # ------------------------------------------------------------------
    # Metodi privati di verifica
    # ------------------------------------------------------------------

    @staticmethod
    def _pair_key(id_a: int, id_b: int) -> tuple[int, int]:
        """
        Restituisce una chiave di coppia ordinata e indipendente dall'ordine.

        Garantisce che la coppia (3, 7) e la coppia (7, 3) producano
        la stessa chiave (3, 7), rendendo il set seen_pairs corretto
        indipendentemente dall'ordine di iterazione.

        Parameters
        ----------
        id_a : int
            Track ID del primo veicolo.
        id_b : int
            Track ID del secondo veicolo.

        Returns
        -------
        tuple[int, int]
            Coppia (min_id, max_id) con il track ID minore sempre per primo.
        """
        if id_a <= id_b:
            return id_a, id_b
        return id_b, id_a

    def _has_overlap(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> tuple[bool, int]:
        """
        Verifica se le maschere di due veicoli si sovrappongono abbastanza.

        Calcola l'area di intersezione pixel-level tra le maschere binarie
        tramite bitwise_AND (geometry.mask_intersection_area) e la confronta
        con la soglia configurata.

        Parameters
        ----------
        vehicle_a : VehicleState
            Stato del primo veicolo (deve avere .mask non-None).
        vehicle_b : VehicleState
            Stato del secondo veicolo (deve avere .mask non-None).

        Returns
        -------
        tuple[bool, int]
            (has_overlap, overlap_area_in_pixels):
              - has_overlap: True se l'area supera mask_overlap_threshold.
              - overlap_area_in_pixels: area effettiva in pixel (0 se nessuna maschera).
        """
        mask_a = vehicle_a.mask
        mask_b = vehicle_b.mask

        # Senza maschera non è possibile calcolare l'overlap pixel-level.
        # Questo accade se la detection ha avuto un poligono degenere.
        if mask_a is None or mask_b is None:
            return False, 0

        # Conta i pixel sovrapposti tra le due maschere binarie uint8.
        overlap_area = int(mask_intersection_area(mask_a, mask_b))

        # Confronto con la soglia: aree sotto la soglia sono considerate
        # rumore o prossimità normale tra veicoli in corsia adiacente.
        return overlap_area >= self.mask_overlap_threshold, overlap_area

    def _both_stopped(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> bool:
        """
        Controlla se entrambi i veicoli risultano fermi da abbastanza frame.

        La condizione "dual stop" è verificata quando:
          - entrambi i veicoli hanno speed_px <= stopped_speed_threshold
          - entrambi hanno stopped_frames >= stopped_frames_threshold

        La doppia condizione (velocità + contatore) riduce i falsi positivi
        causati da singoli frame in cui il tracker perde temporaneamente
        un veicolo in movimento.

        Parameters
        ----------
        vehicle_a : VehicleState
            Stato del primo veicolo.
        vehicle_b : VehicleState
            Stato del secondo veicolo.

        Returns
        -------
        bool
            True se entrambi i veicoli soddisfano la condizione di stop.
        """
        # Accesso diretto agli attributi (VehicleState è una dataclass con
        # slots=True e campi always-present dopo tracker_state.py fix).
        return (
            vehicle_a.stopped_frames >= self.stopped_frames_threshold
            and vehicle_b.stopped_frames >= self.stopped_frames_threshold
            and vehicle_a.speed_px <= self.stopped_speed_threshold
            and vehicle_b.speed_px <= self.stopped_speed_threshold
        )

    def _hard_deceleration(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> bool:
        """
        Controlla se almeno un veicolo sta decelerando bruscamente.

        La frenata brusca è definita da:
            acceleration_px <= strong_deceleration_threshold

        dove ``strong_deceleration_threshold`` è un valore negativo
        (es. -4.0 pixel/frame²). Un'accelerazione inferiore a questa
        soglia indica una frenata improvvisa potenzialmente post-impatto.

        Nota: basta che UNO dei due veicoli deceleri bruscamente per
        attivare la condizione (logica OR): tipicamente nel caso di un
        tamponamento, il veicolo davanti frena bruscamente mentre quello
        dietro continua o rallenta meno.

        Parameters
        ----------
        vehicle_a : VehicleState
            Stato del primo veicolo.
        vehicle_b : VehicleState
            Stato del secondo veicolo.

        Returns
        -------
        bool
            True se almeno un veicolo supera la soglia di decelerazione.
        """
        return (
            vehicle_a.acceleration_px <= self.strong_deceleration_threshold
            or vehicle_b.acceleration_px <= self.strong_deceleration_threshold
        )