import os
import shutil
import random
import matplotlib.pyplot as plt

def dividi_per_strumento(strumento_path, output_path, seed=42):
    random.seed(seed)
    
    # Lista delle tracce
    tracce = [os.path.join(strumento_path, d) for d in os.listdir(strumento_path) if os.path.isdir(os.path.join(strumento_path, d))]
    
    # Contare il numero di immagini in ogni traccia
    tracce_info = []
    for t in tracce:
        num_img = len([f for f in os.listdir(t) if os.path.isfile(os.path.join(t, f))])
        tracce_info.append({'path': t, 'num_img': num_img})
    
    # Ordinare le tracce per numero di immagini (opzionale, migliora il bilanciamento)
    tracce_info.sort(key=lambda x: x['num_img'], reverse=True)
    
    total_images = sum([t['num_img'] for t in tracce_info])
    train_target = total_images * 0.7
    val_target = total_images * 0.15
    test_target = total_images * 0.15
    
    train, val, test = [], [], []
    train_count = val_count = test_count = 0
    
    # Distribuzione greedy delle tracce
    for t in tracce_info:
        if train_count + t['num_img'] <= train_target or len(train) == 0:
            train.append(t)
            train_count += t['num_img']
        elif val_count + t['num_img'] <= val_target or len(val) == 0:
            val.append(t)
            val_count += t['num_img']
        else:
            test.append(t)
            test_count += t['num_img']
    
    # Funzione per copiare le tracce nella destinazione
    def copy_tracce(tracce_list, set_name):
        for t in tracce_list:
            dest_dir = os.path.join(output_path, set_name, os.path.basename(t['path']))
            os.makedirs(dest_dir, exist_ok=True)
            for f in os.listdir(t['path']):
                shutil.copy(os.path.join(t['path'], f), os.path.join(dest_dir, f))
    
    copy_tracce(train, 'train')
    copy_tracce(val, 'val')
    copy_tracce(test, 'test')
    
    print(f"Distribuzione finale immagini: train={train_count}, val={val_count}, test={test_count}")
    
    # Grafico distribuzione
    plt.bar(['train', 'val', 'test'], [train_count, val_count, test_count], color=['green','orange','red'])
    plt.title(f"Distribuzione immagini per {os.path.basename(strumento_path)}")
    plt.ylabel("Numero di immagini")
    plt.show()

# Esempio d'uso:
dividi_per_strumento('./spettrogrammi/violino', 'dataset_diviso/violino')
