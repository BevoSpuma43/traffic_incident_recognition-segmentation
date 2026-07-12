import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # 1. Configurazione percorsi - Sostituisci con il percorso del tuo video
    video_path = "dataset/autostrada.mp4" 
    
    # 2. Caricamento del modello YOLO Segmentation (versione Nano ottimizzata per CPU)
    # Al primo avvio scaricherà automaticamente il file .pt se non presente
    print("[INFO] Caricamento del modello yolo26n-seg.pt...")
    model = YOLO("yolo26n-seg.pt")
    
    # 3. Apertura del file video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire il file video: {video_path}")
        return

    print("[INFO] Avvio riproduzione. Premi 'q' per uscire.")
    
    # ID delle classi COCO relative ai veicoli che vogliamo tracciare
    # 2: auto, 3: moto, 5: autobus, 7: camion
    vehicle_classes = [2, 3, 5, 7]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fine del video o errore di lettura frame.")
            break

        # 4. Inferenza del modello sul singolo frame
        # Usiamo 'persist=True' per attivare il tracker integrato (ByteTrack)
        # Filtriamo le classi usando 'classes' per isolare solo i veicoli
        results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", classes=vehicle_classes, verbose=False)        # Creiamo una copia del frame per disegnare le maschere personalizzate
        overlay = frame.copy()
        
        # Estraiamo il primo risultato (relativo al frame corrente)
        result = results[0]
        
        # Controlliamo se sono state rilevate delle maschere di segmentazione
        if result.masks is not None:
            # I dettagli del tracking (ID dei veicoli assegnati da ByteTrack)
            track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [0] * len(result.masks)
            
            # Iteriamo su ogni maschera rilevata nel frame
            for mask, track_id in zip(result.masks.xy, track_ids):
                # Trasformiamo i punti della maschera in un array NumPy di interi
                points = np.array(mask, dtype=np.int32)
                
                # Generiamo un colore univoco basato sull'ID del veicolo
                # Questo assicura che lo stesso veicolo mantenga lo stesso colore tra i frame
                np.random.seed(track_id)
                color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
                
                # Disegniamo il poligono della maschera pieno sull'immagine di overlay
                cv2.fillPoly(overlay, [points], color)
                
                # Opzionale: calcoliamo il centroide della maschera usando i momenti geometrici
                M = cv2.moments(points)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Disegniamo un pallino al centro del veicolo e l'ID del tracker
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID: {track_id}", (cx - 10, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 5. Effetto Trasparenza (Alpha Blending)
        # Uniamo il frame originale con l'overlay contenente le maschere colorate (trasparenza al 40%)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 6. Mostra il risultato a schermo
        cv2.imshow("YOLOv8 Instance Segmentation - Test", frame)
        
        # Premi 'q' sulla tastiera per interrompere il video prima del tempo
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascio delle risorse
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Risorse rilasciate correttamente.")

if __name__ == "__main__":
    main()