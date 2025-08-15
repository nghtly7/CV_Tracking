import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def analyze_yolo_dataset(dataset_path):
    """
    Analizza un dataset in formato YOLOv8
    :param dataset_path: Percorso base del dataset (con train/val/test)
    """
    results = {}
    sets = ['train', 'valid', 'test']
    class_names = ['Ball', 'Player', 'Referee']  # Modifica se le tue classi hanno nomi diversi
    
    for set_name in sets:
        set_path = os.path.join(dataset_path, set_name)
        labels_dir = os.path.join(set_path, 'labels')
        images_dir = os.path.join(set_path, 'images')
        
        if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
            print(f"⚠️ Directory mancante per {set_name}. Skipping...")
            continue
        
        # Strutture dati per l'analisi
        class_counts = defaultdict(int)
        bbox_sizes = defaultdict(list)
        images_without_objects = 0
        occlusion_stats = defaultdict(int)
        class_confusion = defaultdict(lambda: defaultdict(int))
        resolution_stats = []
        
        # Elabora ogni file di annotazione
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            label_path = os.path.join(labels_dir, label_file)
            image_file = os.path.splitext(label_file)[0] + '.jpg'
            image_path = os.path.join(images_dir, image_file)
            
            if not os.path.exists(image_path):
                # Prova altre estensioni
                for ext in ['.png', '.jpeg', '.JPG']:
                    alt_path = os.path.join(images_dir, os.path.splitext(label_file)[0] + ext)
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
            
            # Carica l'immagine per ottenere dimensioni
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Immagine non trovata: {image_path}")
                continue
                
            img_height, img_width = img.shape[:2]
            resolution_stats.append((img_width, img_height))
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                images_without_objects += 1
                continue
                
            # Processa ogni bounding box
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                class_counts[class_id] += 1
                
                # Converti coordinate YOLO in pixel
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Calcola area e rapporto dimensioni
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                
                bbox_sizes[class_id].append({
                    'width': width,
                    'height': height,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
                
                # Rileva occlusioni (bounding box molto piccole rispetto all'area tipica)
                if class_id == 0:  # Ball
                    if area < 100:  # Soglia per palla occlusa
                        occlusion_stats['ball_occluded'] += 1
                
                # Rileva potenziali confusioni (dimensioni simili tra giocatori e arbitri)
                elif class_id in [1, 2]:  # Player o Referee
                    if 0.8 < aspect_ratio < 1.2 and 5000 < area < 15000:
                        class_confusion[class_id]['square_medium'] += 1
                    
        # Calcolo delle statistiche
        set_stats = {
            'total_images': len(os.listdir(images_dir)),
            'labeled_images': len(os.listdir(labels_dir)),
            'images_without_objects': images_without_objects,
            'class_distribution': dict(class_counts),
            'avg_bbox_size': {},
            'occlusion_stats': dict(occlusion_stats),
            'confusion_risk': dict(class_confusion),
            'resolution_variation': np.std(resolution_stats, axis=0).tolist() if resolution_stats else [0, 0]
        }
        
        # Calcola dimensioni medie per classe
        for class_id, sizes in bbox_sizes.items():
            if sizes:
                avg_width = np.mean([s['width'] for s in sizes])
                avg_height = np.mean([s['height'] for s in sizes])
                avg_area = np.mean([s['area'] for s in sizes])
                set_stats['avg_bbox_size'][class_id] = {
                    'width': avg_width,
                    'height': avg_height,
                    'area': avg_area
                }
        
        results[set_name] = set_stats
    
    return results, class_names

def generate_report(results, class_names):
    """Genera un report dettagliato con grafici"""
    report = "[ANALYSIS] YOLOv8 Dataset Analysis Report\n\n"
    
    for set_name, stats in results.items():
        report += f"=== {set_name.upper()} SET ===\n"
        report += f"- Images totali: {stats['total_images']}\n"
        report += f"- Images annotate: {stats['labeled_images']}\n"
        report += f"- Images senza oggetti: {stats['images_without_objects']}\n"
        report += f"- Variazione risoluzione (dev std): W={stats['resolution_variation'][0]:.1f}, H={stats['resolution_variation'][1]:.1f}\n\n"
        
        report += "[CLASSES] Distribuzione classi:\n"
        for class_id, count in stats['class_distribution'].items():
            report += f"  {class_names[class_id]}: {count} bbox ({count/stats['labeled_images']:.1f} per image)\n"
        
        report += "\n[SIZES] Dimensioni medie bbox (pixels):\n"
        for class_id, size_info in stats['avg_bbox_size'].items():
            report += (f"  {class_names[class_id]}: "
                       f"W={size_info['width']:.1f}px, "
                       f"H={size_info['height']:.1f}px, "
                       f"Area={size_info['area']:.1f}px²\n")
        
        if 'ball_occluded' in stats['occlusion_stats']:
            report += (f"\n[WARNING] Palle occluse: {stats['occlusion_stats']['ball_occluded']} "
                       f"({stats['occlusion_stats']['ball_occluded']/stats['class_distribution'].get(0,1)*100:.1f}% delle palle)\n")
        
        report += "\n[CONFUSION] Rischi confusione:\n"
        for class_id, conf_stats in stats['confusion_risk'].items():
            for conf_type, count in conf_stats.items():
                report += f"  {class_names[class_id]} - {conf_type}: {count} bbox\n"
        
        report += "\n" + "-"*50 + "\n"
    
    # Visualizzazione grafica
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Distribuzione classi
    for set_name, stats in results.items():
        class_dist = [stats['class_distribution'].get(i, 0) for i in range(len(class_names))]
        axs[0].bar(np.arange(len(class_names)) + 0.2 * list(results.keys()).index(set_name), 
                   class_dist, 
                   width=0.2, 
                   label=set_name)
    axs[0].set_title('Distribuzione Classi')
    axs[0].set_xticks(range(len(class_names)))
    axs[0].set_xticklabels(class_names)
    axs[0].legend()
    
    # Dimensioni bbox
    for i, class_name in enumerate(class_names):
        sizes = []
        for set_name in results:
            if i in results[set_name]['avg_bbox_size']:
                sizes.append(results[set_name]['avg_bbox_size'][i]['area'])
        if sizes:
            axs[1].bar([class_name] * len(sizes), sizes, label=[f"{s} set" for s in results.keys()])
    axs[1].set_title('Area Media BBox (px²)')
    axs[1].set_yscale('log')
    
    # Risoluzioni
    all_res = []
    for set_name, stats in results.items():
        if 'resolution_variation' in stats:
            all_res.append(stats['resolution_variation'])
    if all_res:
        axs[2].bar(results.keys(), [r[0] for r in all_res], label='Larghezza')
        axs[2].bar(results.keys(), [r[1] for r in all_res], bottom=[r[0] for r in all_res], label='Altezza')
        axs[2].set_title('Variazione Risoluzione (Dev. Std)')
        axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    #plt.show()
    
    return report

# ===== USAGE =====
if __name__ == "__main__":
    dataset_path = "complete_dataset_yolov8/"  # Modifica con il tuo percorso
    
    print("🚀 Starting dataset analysis...")
    results, class_names = analyze_yolo_dataset(dataset_path)
    report = generate_report(results, class_names)
    
    print(report)
    with open('dataset_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Analysis complete! Check:")
    print("- dataset_report.txt")
    print("- dataset_analysis.png")