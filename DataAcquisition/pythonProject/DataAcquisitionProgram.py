import cv2
import os
import re
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# import numpy as np
# import random

# Define global variables
window_closed = False
user_name = ""
user_letter = ""
capture_requested = False
quit_requested = False
capture_count = 0


# Funzione per trovare il prossimo numero di cattura disponibile
def get_next_capture_number(directory, name, letter):
    existing_files = os.listdir(directory)
    max_number = 0
    pattern = re.compile(rf"{letter}_{name}_(\d+)\.jpg")
    for file in existing_files:
        match = pattern.match(file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
    return max_number + 1


# Funzione per applicare data augmentation
'''
def augment_image(image, w, h):
    # Flip orizzontale
    if random.choice([True, False]):
        image = cv2.flip(image, 1)

    # Rotazione
    if random.choice([True, False]):
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))

    # Traslazione
    if random.choice([True, False]):
        x_shift = random.randint(-10, 10)
        y_shift = random.randint(-10, 10)
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        image = cv2.warpAffine(image, M, (w, h))

    # Cambio luminosità
    if random.choice([True, False]):
        value = random.randint(-30, 30)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return image
'''


# Funzione per salvare l'immagine
def save_image(image, filename):
    resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite(filename, resized_image)
    log_message(f'Immagine salvata come {filename}')


# Funzione per gestire i click dei pulsanti
def capture_image():
    global capture_requested
    capture_requested = True


def close_window():
    global quit_requested
    quit_requested = True


# Funzione per loggare messaggi nella finestra
def log_message(message):
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, message + "\n")
    log_text.config(state=tk.DISABLED)
    log_text.see(tk.END)


# Funzione principale per la cattura delle immagini
def capture_images(name, letter):
    global capture_requested, quit_requested, capture_count, log_text
    capture_requested = False
    quit_requested = False

    # Crea la cartella per il dataset se non esiste
    dataset_dir = os.path.join("Dataset_Elvio_esteso_black_background", letter)
    os.makedirs(dataset_dir, exist_ok=True)

    # Numero di cattura
    capture_count = get_next_capture_number(dataset_dir, name, letter)

    # Dimensione dell'immagine ritagliata
    crop_size = 256

    # Apri la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam")
        return

    print("Usa i pulsanti per catturare l'immagine o chiudere la finestra.")

    # Configura la finestra Tkinter
    root = tk.Tk()
    root.title("Acquisizione")
    root.protocol("WM_DELETE_WINDOW", close_window)

    # Dimensioni della finestra
    window_width = 800
    window_height = 600

    # Centra la finestra di acquisizione sullo schermo
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    # Canvas per visualizzare il feed della webcam
    canvas = tk.Canvas(root, width=640, height=480)
    canvas.pack()

    # Frame per posizionare i pulsanti in basso al centro
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, pady=10)

    # Aggiungi i pulsanti
    btn_capture = tk.Button(button_frame, text="Acquisisci (c)", command=capture_image)
    btn_capture.pack(side=tk.LEFT, padx=20)
    btn_close = tk.Button(button_frame, text="Chiudi (q)", command=close_window)
    btn_close.pack(side=tk.RIGHT, padx=20)

    # Text widget per loggare messaggi
    log_text = tk.Text(root, height=10, state=tk.DISABLED)
    log_text.pack(side=tk.BOTTOM, fill=tk.X)

    def update_frame():
        global capture_requested, quit_requested, capture_count
        ret, frame = cap.read()
        if not ret:
            log_message("Errore: Impossibile acquisire l'immagine")
            return

        # Dimensioni del frame
        h, w, _ = frame.shape

        # Coordinate del centro del frame
        center_x, center_y = w // 2, h // 2

        # Coordinate del riquadro di cattura
        x1, y1 = center_x - crop_size // 2, center_y - crop_size // 2
        x2, y2 = center_x + crop_size // 2, center_y + crop_size // 2

        if capture_requested:
            # Cattura l'immagine alla risoluzione normale senza il bordo verde
            capture_img = frame[y1:y2, x1:x2]
            # capture_img = augment_image(capture_img, x2 - x1, y2 - y1)  # Applica data augmentation
            filename = os.path.join(dataset_dir, f"{letter}_{name}_{capture_count}.jpg")
            save_image(capture_img, filename)
            capture_count += 1
            capture_requested = False  # Reset della richiesta di cattura

        # Disegna un riquadro verde al centro del frame (dopo la cattura dell'immagine)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Converti il frame in un'immagine Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)

        # Aggiorna il canvas con la nuova immagine
        canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
        canvas.image = frame_tk

        if not quit_requested:
            root.after(5, update_frame)
        else:
            cap.release()
            root.destroy()

    update_frame()

    # Mappa i tasti "c" e "q" per i pulsanti corrispondenti
    root.bind('<c>', lambda event: capture_image())
    root.bind('<q>', lambda event: close_window())

    log_message("Usa i pulsanti per catturare l'immagine o chiudere la finestra.")

    root.mainloop()


# Funzione per creare la finestra di dialogo personalizzata
def get_user_input():
    def on_submit(event=None):
        global user_name, user_letter
        user_name = name_entry.get().upper()
        user_letter = letter_entry.get().lower()

        # Controlli di validità
        if not user_name:
            messagebox.showerror("Errore", "Il nome non può essere vuoto.")
            return
        if user_name.isdigit():
            messagebox.showerror("Errore", "Il nome non può essere un numero.")
            return
        if not user_letter or len(user_letter) != 1:
            messagebox.showerror("Errore", "Il carattere deve essere esattamente uno.")
            return
        if user_letter in ['g', 's', 'j', 'z']:
            messagebox.showerror("Errore", "Il carattere inserito non è consentito.\n"
                                           "I caratteri G, S, J, Z sono esclusi.")
            return
        if user_letter.isdigit():
            messagebox.showerror("Errore", "Il carattere non può essere un numero.")
            return

        dialog.destroy()

    def on_close():
        global window_closed
        window_closed = True
        dialog.destroy()

    dialog = tk.Tk()
    dialog.title("Inserisci le informazioni")

    # Centra la finestra di dialogo sullo schermo
    window_width = 400
    window_height = 300
    screen_width = dialog.winfo_screenwidth()
    screen_height = dialog.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    dialog.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    tk.Label(dialog, text="Nome:", font=("Arial", 14)).pack(pady=10)
    name_entry = tk.Entry(dialog, font=("Arial", 14))
    name_entry.pack(pady=10)

    tk.Label(dialog, text="Carattere della LIS:", font=("Arial", 14)).pack(pady=10)
    letter_entry = tk.Entry(dialog, font=("Arial", 14))
    letter_entry.pack(pady=10)

    submit_button = tk.Button(dialog, text="Submit", font=("Arial", 14), command=on_submit)
    submit_button.pack(pady=20)

    dialog.bind('<Return>', on_submit)
    dialog.protocol("WM_DELETE_WINDOW", on_close)

    dialog.mainloop()


# Funzione per avviare l'interfaccia utente
def start_ui():
    global window_closed
    window_closed = False
    get_user_input()
    if not window_closed:
        capture_images(user_name, user_letter)


if __name__ == "__main__":
    start_ui()
