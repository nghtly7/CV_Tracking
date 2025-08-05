from ultralytics import YOLO
import cv2
import os
# Carica il modello addestrato
model = YOLO("best_nano.pt")  # Cambia il path se necessario

# Esegui inferenza
# Crea la cartella di output se non esiste
os.makedirs("frames/labeled", exist_ok=True)

# Esegui inferenza su tutte le immagini della cartella
results = model(
    source="frames/out4/",
    save=True,
    project="frames/labeled", # Salva risultati qui dentro  
    name="out4",   # Sottocartella dentro project/
    conf=0.5,            # Confidence threshold
    imgsz=640            # (opzionale) Resize delle immagini prima dell’inferenza
)


