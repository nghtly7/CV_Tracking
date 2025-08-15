import os
import glob
from pathlib import Path

new =['Ball', 'Player', 'Referee']
current = [
    'Ball',        # 0 → 0 (Ball)
    'Red_0',       # 1 → 1 (Player)
    'Red_11',      # 2 → 1 (Player)
    'Red_12',      # 3 → 1 (Player)
    'Red_16',      # 4 → 1 (Player)
    'Red_2',       # 5 → 1 (Player)
    'Refree_F',    # 6 → 2 (Referee)
    'Refree_M',    # 7 → 2 (Referee)
    'White_13',    # 8 → 1 (Player)
    'White_16',    # 9 → 1 (Player)
    'White_25',    # 10 → 1 (Player)
    'White_27',    # 11 → 1 (Player)
    'White_34'     # 12 → 1 (Player)
]

def correct_label_file(file_path):
    """
    Corregge le etichette di classe in un singolo file di annotazione YOLO
    
    Args:
        file_path: Percorso del file .txt da correggere
    
    Returns:
        int: Numero di linee modificate
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        modified_lines = 0
        corrected_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:  # Salta linee vuote
                corrected_lines.append(line + '\n')
                continue
            
            parts = line.split()
            if len(parts) < 5:  # Le annotazioni YOLO devono avere almeno 5 valori
                corrected_lines.append(line + '\n')
                continue
            
            # Estrai l'ID della classe (primo valore)
            try:
                class_id = int(parts[0])
            except ValueError:
                corrected_lines.append(line + '\n')
                continue
            
            # Applica le regole di conversione
            new_class_id = class_id
            
            if class_id == 0:
                # Ball rimane 0
                new_class_id = 0
            elif class_id in {1, 2, 3, 4, 5, 8, 9, 10, 11, 12}:
                # Tutti i giocatori diventano 1 (Player)
                new_class_id = 1
            elif class_id in {6, 7}:
                # Arbitri diventano 2 (Referee)
                new_class_id = 2
            
            # Se l'ID è cambiato, aggiorna la linea
            if new_class_id != class_id:
                parts[0] = str(new_class_id)
                modified_lines += 1
                print(f"  Cambiato: classe {class_id} → {new_class_id}")
            
            # Ricostruisci la linea
            corrected_lines.append(' '.join(parts) + '\n')
        
        # Scrivi il file corretto
        if modified_lines > 0:
            with open(file_path, 'w') as f:
                f.writelines(corrected_lines)
        
        return modified_lines
    
    except Exception as e:
        print(f"Errore nel processare {file_path}: {e}")
        return 0

def process_labels_directory(labels_dir):
    """
    Processa tutti i file .txt in una directory di etichette
    
    Args:
        labels_dir: Percorso della directory contenente i file .txt
    
    Returns:
        tuple: (numero di file processati, numero totale di linee modificate)
    """
    if not os.path.exists(labels_dir):
        print(f"Directory non trovata: {labels_dir}")
        return 0, 0
    
    # Trova tutti i file .txt nella directory
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not txt_files:
        print(f"Nessun file .txt trovato in: {labels_dir}")
        return 0, 0
    
    print(f"Trovati {len(txt_files)} file .txt in {labels_dir}")
    
    total_files_modified = 0
    total_lines_modified = 0
    
    for txt_file in txt_files:
        print(f"Processando: {os.path.basename(txt_file)}")
        lines_modified = correct_label_file(txt_file)
        
        if lines_modified > 0:
            total_files_modified += 1
            total_lines_modified += lines_modified
            print(f"  → {lines_modified} etichette corrette")
        else:
            print(f"  → Nessuna modifica necessaria")
    
    return total_files_modified, total_lines_modified

def correct_all_labels(dataset_path="complete_dataset_yolov8"):
    """
    Corregge tutte le etichette nel dataset YOLO
    
    Args:
        dataset_path: Percorso del dataset
    """
    print("=== CORREZIONE ETICHETTE DATASET YOLO ===")
    print(f"Regole di conversione:")
    print(f"  0 (Ball) → 0 (Ball)")
    print(f"  {{1,2,3,4,5,8,9,10,11,12}} (Players) → 1 (Player)")
    print(f"  {{6,7}} (Referees) → 2 (Referee)")
    print()
    
    # Directory delle etichette da processare
    label_directories = [
        os.path.join(dataset_path, "train", "labels"),
        os.path.join(dataset_path, "test", "labels"),
        os.path.join(dataset_path, "valid", "labels")
    ]
    
    total_files_corrected = 0
    total_labels_corrected = 0
    
    for labels_dir in label_directories:
        print(f"Processando directory: {labels_dir}")
        files_modified, lines_modified = process_labels_directory(labels_dir)
        total_files_corrected += files_modified
        total_labels_corrected += lines_modified
        print(f"Risultato: {files_modified} file modificati, {lines_modified} etichette corrette")
        print()
    
    print("=== RIEPILOGO ===")
    print(f"File totali modificati: {total_files_corrected}")
    print(f"Etichette totali corrette: {total_labels_corrected}")
    print("Correzione completata!")

if __name__ == "__main__":
    print("Inizio correzione etichette...")
    correct_all_labels()