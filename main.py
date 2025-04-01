import cv2
import dlib
import os
import uuid
import numpy as np
import tkinter as tk
from deepface import DeepFace
from PIL import Image, ImageTk

def detectar_acessorios(landmarks):
    """Identifica se a pessoa está usando óculos ou boné."""
    usando_oculos = landmarks.part(36).y < landmarks.part(39).y  # Posição dos olhos
    usando_bone = landmarks.part(24).y < landmarks.part(27).y  # Teste da posição da testa
    
    if usando_oculos and usando_bone:
        return "Boné e Óculos"
    elif usando_oculos:
        return "Óculos"
    elif usando_bone:
        return "Boné"
    return "Sem acessórios"

def salvar_imagem(frame):
    """Salva a imagem no diretório com um nome único."""
    filename = f"reconhecimento_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Reconhecimento válido. Imagem salva como {filename}")

def atualizar_camera():
    ret, frame = cap.read()
    if not ret:
        return
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        landmarks = predictor(gray, face)
        acessorios = detectar_acessorios(landmarks)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, acessorios, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        rosto = frame[y:y+h, x:x+w]
        try:
            analysis = DeepFace.analyze(rosto, actions=["age"], enforce_detection=False)
            if analysis:
                salvar_imagem(frame)
        except:
            pass
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    camera_label.configure(image=img)
    camera_label.image = img
    root.after(10, atualizar_camera)

# Configuração do Tkinter
root = tk.Tk()
root.title("Validação Facial")
root.geometry("400x600")
root.configure(bg="#f8f9f4")

titulo = tk.Label(root, text="Validação facial", font=("Arial", 14, "bold"), bg="#f8f9f4")
titulo.pack()

instrucao = tk.Label(root, text="Encaixe seu rosto no formato e clique no botão abaixo", font=("Arial", 10), bg="#f8f9f4")
instrucao.pack()

canvas = tk.Canvas(root, width=300, height=400, bg="#f8f9f4", highlightthickness=0)
canvas.pack()

# Criar formato oval (máscara)
canvas.create_oval(50, 50, 250, 350, outline="purple", width=3)

camera_label = tk.Label(root, bg="#f8f9f4")
camera_label.pack()

botao_ok = tk.Button(root, text="✔", font=("Arial", 14), bg="purple", fg="white", command=lambda: print("Validado"))
botao_ok.pack(side=tk.LEFT, padx=50, pady=20)

botao_cancel = tk.Button(root, text="✖", font=("Arial", 14), bg="red", fg="white", command=root.quit)
botao_cancel.pack(side=tk.RIGHT, padx=50, pady=20)

# Inicializar câmera e modelo
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
atualizar_camera()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
