import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import numpy as np

alph = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'h',
    7: 'i',
    8: 'k',
    9: 'l',
    10: 'm',
    11: 'n',
    12: 'o',
    13: 'p',
    14: 'q',
    15: 'r',
    16: 't',
    17: 'u',
    18: 'v',
    19: 'w',
    20: 'x',
    21: 'y',
}

# Load the pre-trained model
model = torch.load("TrainedModels/SignLanguage_DeepCNN_WithDropout_WithDA_LessParams_lr_001_mom99_ep20_elvio.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define image resolution
imageResolution = (64, 64)

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(imageResolution),
    transforms.ToTensor(),
])

# List to store characters
characters = []

gesture_to_alphabet = None


def get_center_coordinates(frame, size=350):
    """Utility function to get the center square of the frame"""
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    x, y = center[0] - size // 2, center[1] - size // 2
    return x, y, size


def detect_gesture(frame, input_model):
    """ Detects the gesture in the center square of the frame with the trained model """
    x, y, size = get_center_coordinates(frame)
    hand_img = frame[y:y + size, x:x + size]
    hand_img = transform(Image.fromarray(hand_img)).to(device)
    hand_img = hand_img.unsqueeze(0)
    with torch.no_grad():
        output = input_model(hand_img)
    gesture = output.argmax(dim=1).item()
    return gesture, hand_img


def update_frame():
    """Updates the frame with the gesture prediction, if the hand is in the center, and displays it in the Tkinter window.
        Also displays the currently predicted gesture.
        Updates the global variable that stores the current gesture.
    """
    ret, frame = cap.read()
    if not ret:
        return

    # frame = cv2.flip(frame, 1)

    # Draw green square in the center
    x, y, size = get_center_coordinates(frame)
    cv2.rectangle(frame, (x-5, y-5), (x + size, y + size), (0, 255, 0), 1)

    # Check if the hand is in the center square and predict gesture using the trained model
    gesture, hand_img = detect_gesture(frame, model)

    # Display the gesture on the frame
    global gesture_to_alphabet
    gesture_to_alphabet = alph[gesture]
    gesture_text = f"Gesture: {gesture_to_alphabet}"
    cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Convert frame to Image for Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the label with the new frame
    label.imgtk = img_tk
    label.config(image=img_tk)

    label.after(16, update_frame) # 16ms => 60 FPS

    global current_frame
    current_frame = frame


def add_character(event):
    """Event function to add the currently predicted gesture to the list of characters. Triggered by SPACEBAR"""
    characters.append(gesture_to_alphabet)
    char_list_label.config(text="".join(map(str, characters)))

def clear_last_input(event):
    """Event function to clear the last character from the list of characters"""
    if characters:
        characters.pop()
        char_list_label.config(text="".join(map(str, characters)))


def quit_program(event):
    """Event function to quit the program. Triggered by ESCAPE"""
    cap.release()
    cv2.destroyAllWindows()
    root.quit()


def main():
    global root, label, hand_img_label, char_list_label, cap
    # Setup Tkinter
    root = tk.Tk()
    root.title("Sign Language Recognition | 'SPACE' to add character, 'ESC' to quit")

    label = Label(root)
    label.pack()

    hand_img_label = Label(root)
    hand_img_label.pack(side=tk.RIGHT, padx=10, pady=10)

    char_list_label = Label(root, text="", font=('Helvetica', 18))
    char_list_label.pack()

    # Bind events
    root.bind("<space>", add_character)
    root.bind("<Escape>", quit_program)
    root.bind("<BackSpace>", clear_last_input)

    # Get the camera feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Main loop
    update_frame()
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
