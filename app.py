from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
import uvicorn
import cv2
import uuid
import os
import dlib
from typing import List, Dict, Optional
from pydantic import BaseModel

# Cria diretório para salvar imagens temporárias
os.makedirs("temp", exist_ok=True)
os.makedirs("database", exist_ok=True)

app = FastAPI(title="API de Reconhecimento Facial")

# Base de dados em memória para armazenar codificações faciais
face_database = {}

# Carrega o detector de pontos faciais do dlib para uso na detecção de acessórios
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(predictor_path)
    predictor_loaded = True
except:
    predictor_loaded = False
    print(f"AVISO: O arquivo {predictor_path} não foi encontrado. A detecção de acessórios estará limitada.")
    print(f"Para habilitar a detecção completa, baixe o arquivo em: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("Descompacte e coloque-o no mesmo diretório que este script.")

class PersonModel(BaseModel):
    person_id: str
    name: str

class RecognitionResult(BaseModel):
    person_id: Optional[str] = None
    name: Optional[str] = None
    confidence: float
    recognized: bool

class AccessoryDetectionResult(BaseModel):
    has_glasses: bool
    has_hat: bool
    is_acceptable: bool
    message: str
    debug_info: Optional[Dict] = None

def detect_accessories(image_path, debug_mode=False):
    """Detecta se a pessoa está usando óculos ou boné/chapéu com parâmetros ajustados."""
    image = cv2.imread(image_path)
    if image is None:
        return {
            "has_glasses": False,
            "has_hat": False,
            "is_acceptable": False,
            "message": "Erro ao ler a imagem. Formato não suportado ou arquivo corrompido.",
            "debug_info": {}
        }
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resultado padrão
    result = {
        "has_glasses": False,
        "has_hat": False,
        "is_acceptable": True,
        "message": "Imagem aceita.",
        "debug_info": {} if debug_mode else None
    }
    
    # Detecta faces
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        result["is_acceptable"] = False
        result["message"] = "Nenhuma face detectada na imagem."
        return result
    
    # Inicializa variáveis para debug
    if debug_mode:
        result["debug_info"] = {
            "edge_counts": {},
            "thresholds": {
                "glasses_edge_threshold": 350,  # Aumentado de 200 para evitar falsos positivos
                "forehead_ratio_threshold": 0.12  # Reduzido para ser mais tolerante
            },
            "measurements": {}
        }
    
    # Método 2: Tenta usar dlib para detecção mais avançada se estiver disponível
    if predictor_loaded:
        dlib_rects = detector(gray, 1)
        
        if len(dlib_rects) > 0:
            # Usa o preditor de 68 pontos para obter landmarks faciais detalhados
            shape = predictor(gray, dlib_rects[0])
            
            # Extrai pontos relevantes
            points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
            
            # Verifica óculos (pontos ao redor dos olhos)
            # Amplia a região dos olhos para melhor detecção
            eye_region = gray[
                max(0, np.min(points[36:48, 1])-15):
                min(gray.shape[0], np.max(points[36:48, 1])+15),
                max(0, np.min(points[36:48, 0])-15):
                min(gray.shape[1], np.max(points[36:48, 0])+15)
            ]
            
            if eye_region.size > 0:
                # Usa Sobel ao invés de Canny para melhor detecção de bordas em óculos
                sobelx = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
                abs_sobelx = np.absolute(sobelx)
                abs_sobely = np.absolute(sobely)
                sobel_mag = np.sqrt(abs_sobelx**2 + abs_sobely**2)
                sobel_mag = np.uint8(sobel_mag * 255 / np.max(sobel_mag))
                
                # Aplica um limiar para destacar apenas bordas fortes
                _, sobel_thresh = cv2.threshold(sobel_mag, 100, 255, cv2.THRESH_BINARY)
                
                # Conta bordas fortes para detectar armações de óculos
                edge_count = np.sum(sobel_thresh > 0)
                
                if debug_mode:
                    result["debug_info"]["edge_counts"]["eye_region"] = int(edge_count)
                    result["debug_info"]["edge_counts"]["eye_region_size"] = eye_region.size
                    if eye_region.size > 0:
                        result["debug_info"]["edge_counts"]["edge_density"] = float(edge_count) / eye_region.size
                
                # Ajustado para ser menos sensível (limiar mais alto)
                # Considera também o tamanho da região para normalizar
                edge_density = float(edge_count) / eye_region.size if eye_region.size > 0 else 0
                glasses_threshold = 0.15  # Valor ajustável - mais alto = menos sensível
                
                if edge_density > glasses_threshold:
                    result["has_glasses"] = True
            
            # Verifica boné/chapéu (altura da testa)
            # Mede a proporção entre a altura da testa e a altura total do rosto
            forehead_top = np.min(points[0:27, 1])
            eyebrow_top = np.min(points[17:27, 1])
            face_height = np.max(points[:, 1]) - forehead_top
            forehead_height = eyebrow_top - forehead_top
            
            if face_height > 0:
                forehead_ratio = forehead_height / face_height
                
                if debug_mode:
                    result["debug_info"]["measurements"]["forehead_height"] = float(forehead_height)
                    result["debug_info"]["measurements"]["face_height"] = float(face_height)
                    result["debug_info"]["measurements"]["forehead_ratio"] = float(forehead_ratio)
                
                # Se a testa for muito curta em relação ao rosto, pode indicar um chapéu/boné
                # Ajustado para ser menos sensível
                hat_threshold = 0.12  # Valor ajustável - mais baixo = menos sensível
                if forehead_ratio < hat_threshold and forehead_height < 20:
                    result["has_hat"] = True
    else:
        # Método alternativo usando apenas face_recognition se dlib não estiver disponível
        # Este método é menos preciso, mas fornece uma alternativa
        
        # Abordagem simplificada para detecção de óculos
        face_landmarks = face_recognition.face_landmarks(image)
        
        if face_landmarks and len(face_landmarks) > 0:
            landmarks = face_landmarks[0]
            
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                left_eye = landmarks['left_eye']
                right_eye = landmarks['right_eye']
                
                # Extrai região dos olhos com margem maior
                eye_region = gray[
                    max(0, min(p[1] for p in left_eye + right_eye) - 20):
                    min(gray.shape[0], max(p[1] for p in left_eye + right_eye) + 20),
                    max(0, min(p[0] for p in left_eye + right_eye) - 20):
                    min(gray.shape[1], max(p[0] for p in left_eye + right_eye) + 20)
                ]
                
                if eye_region.size > 0:
                    # Usa Sobel para detecção de bordas
                    sobelx = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
                    abs_sobelx = np.absolute(sobelx)
                    abs_sobely = np.absolute(sobely)
                    sobel_mag = np.sqrt(abs_sobelx**2 + abs_sobely**2)
                    
                    if np.max(sobel_mag) > 0:  # Evita divisão por zero
                        sobel_mag = np.uint8(sobel_mag * 255 / np.max(sobel_mag))
                        _, sobel_thresh = cv2.threshold(sobel_mag, 100, 255, cv2.THRESH_BINARY)
                        edge_count = np.sum(sobel_thresh > 0)
                        
                        if debug_mode:
                            result["debug_info"]["edge_counts"]["eye_region_alt"] = int(edge_count)
                            result["debug_info"]["edge_counts"]["eye_region_alt_size"] = eye_region.size
                        
                        # Normaliza pela área para ser consistente com diferentes tamanhos de imagem
                        edge_density = float(edge_count) / eye_region.size
                        glasses_threshold = 0.15  # Ajustado para evitar falsos positivos
                        
                        if edge_density > glasses_threshold:
                            result["has_glasses"] = True
            
            # Detecção simplificada de boné/chapéu
            # Normalmente, o topo da cabeça não é bem detectado por face_recognition,
            # então esta é uma estimativa aproximada baseada na proporção do rosto
            
            if 'chin' in landmarks and 'nose_bridge' in landmarks:
                chin_y = max(p[1] for p in landmarks['chin'])
                nose_top_y = min(p[1] for p in landmarks['nose_bridge'])
                face_height = chin_y - nose_top_y
                
                # Região acima do nariz
                top, right, bottom, left = face_locations[0]
                expected_forehead = int(face_height * 0.6)  # Estimativa da altura da testa
                
                # Se a região detectada acima do nariz for muito menor que o esperado
                if (nose_top_y - top) < (expected_forehead * 0.3):  # 30% do esperado
                    result["has_hat"] = True
    
    # Determina se a imagem é aceita
    if result["has_glasses"] or result["has_hat"]:
        result["is_acceptable"] = False
        accessories = []
        if result["has_glasses"]:
            accessories.append("óculos")
        if result["has_hat"]:
            accessories.append("boné/chapéu")
        
        result["message"] = f"Detectamos {' e '.join(accessories)}. Por favor, envie uma nova foto sem esses acessórios."
    
    return result

def process_image(image_path):
    """Processa a imagem e retorna as codificações faciais."""
    # Carrega a imagem usando face_recognition
    image = face_recognition.load_image_file(image_path)
    
    # Encontra todas as faces na imagem
    face_locations = face_recognition.face_locations(image)
    
    # Se nenhuma face for encontrada, retorna erro
    if len(face_locations) == 0:
        return None
    
    # Obter codificações faciais
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    return face_encodings, face_locations

@app.post("/check-accessories/", response_model=AccessoryDetectionResult)
async def check_accessories(file: UploadFile = File(...), debug: bool = False):
    """
    Verifica se a pessoa na imagem está usando óculos ou boné/chapéu.
    - **file**: Imagem contendo a face para verificação
    - **debug**: Se verdadeiro, retorna informações adicionais para diagnóstico
    """
    # Salva a imagem temporariamente
    temp_id = str(uuid.uuid4())
    temp_file_path = f"temp/{temp_id}.jpg"
    
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Verifica acessórios
        result = detect_accessories(temp_file_path, debug_mode=debug)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao detectar acessórios: {str(e)}")
        
    finally:
        # Limpa arquivos temporários
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/register/", response_model=PersonModel)
async def register_face(name: str, file: UploadFile = File(...), check_accessories: bool = True):
    """
    Registra uma nova face no sistema.
    - **name**: Nome da pessoa
    - **file**: Imagem contendo a face
    - **check_accessories**: Se verdadeiro, verifica se a pessoa está usando óculos ou boné (padrão: True)
    """
    # Gera um ID único para a pessoa
    person_id = str(uuid.uuid4())
    
    # Salva a imagem temporariamente
    temp_file_path = f"temp/{person_id}.jpg"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Verifica acessórios se solicitado
        if check_accessories:
            accessory_result = detect_accessories(temp_file_path)
            if not accessory_result["is_acceptable"]:
                os.remove(temp_file_path)
                raise HTTPException(status_code=400, detail=accessory_result["message"])
        
        # Processa a imagem
        result = process_image(temp_file_path)
        if result is None:
            os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail="Nenhuma face detectada na imagem")
            
        face_encodings, _ = result
        
        # Salva a codificação facial
        face_database[person_id] = {
            "name": name,
            "encoding": face_encodings[0].tolist()  # Converte para lista para permitir serialização
        }
        
        # Salva a imagem na base de dados permanente
        os.rename(temp_file_path, f"database/{person_id}.jpg")
        
        return {"person_id": person_id, "name": name}
    
    except HTTPException:
        raise
    
    except Exception as e:
        # Limpa arquivos temporários em caso de erro
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")

@app.post("/recognize/", response_model=List[RecognitionResult])
async def recognize_face(file: UploadFile = File(...), threshold: float = 0.6, check_accessories: bool = True):
    """
    Reconhece faces em uma imagem.
    - **file**: Imagem contendo faces para reconhecimento
    - **threshold**: Limiar de confiança (padrão: 0.6, menor valor = maior sensibilidade)
    - **check_accessories**: Se verdadeiro, verifica se a pessoa está usando óculos ou boné (padrão: True)
    """
    # Salva a imagem temporariamente
    temp_id = str(uuid.uuid4())
    temp_file_path = f"temp/{temp_id}.jpg"
    
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Verifica acessórios se solicitado
        if check_accessories:
            accessory_result = detect_accessories(temp_file_path)
            if not accessory_result["is_acceptable"]:
                raise HTTPException(status_code=400, detail=accessory_result["message"])
        
        # Processa a imagem
        result = process_image(temp_file_path)
        if result is None:
            raise HTTPException(status_code=400, detail="Nenhuma face detectada na imagem")
            
        unknown_encodings, _ = result
        
        # Lista para armazenar resultados de reconhecimento
        recognition_results = []
        
        # Para cada face encontrada na imagem
        for unknown_encoding in unknown_encodings:
            best_match = {"person_id": None, "name": None, "confidence": 0, "recognized": False}
            
            # Compara com todas as faces no banco de dados
            for person_id, data in face_database.items():
                # Calcula a distância entre faces (menor = mais similar)
                known_encoding = np.array(data["encoding"])
                face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
                
                # Converte distância para confiança (0-1)
                confidence = 1 - face_distance
                
                # Se confiança maior que threshold e melhor que o atual
                if confidence > threshold and confidence > best_match["confidence"]:
                    best_match = {
                        "person_id": person_id,
                        "name": data["name"],
                        "confidence": float(confidence),
                        "recognized": True
                    }
            
            # Se nenhuma correspondência for encontrada acima do limiar
            if not best_match["recognized"]:
                best_match = {
                    "person_id": None,
                    "name": None,
                    "confidence": 0.0,
                    "recognized": False
                }
                
            recognition_results.append(best_match)
        
        return recognition_results
        
    except HTTPException:
        raise
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar reconhecimento: {str(e)}")
        
    finally:
        # Limpa arquivos temporários
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/database/", response_model=Dict[str, Dict])
async def get_database():
    """Lista todas as pessoas cadastradas no banco de dados."""
    result = {}
    for person_id, data in face_database.items():
        result[person_id] = {"name": data["name"]}
    return result

@app.delete("/database/{person_id}")
async def delete_person(person_id: str):
    """Remove uma pessoa do banco de dados."""
    if person_id not in face_database:
        raise HTTPException(status_code=404, detail="Pessoa não encontrada")
    
    # Remove do banco de dados em memória
    del face_database[person_id]
    
    # Remove arquivo de imagem
    if os.path.exists(f"database/{person_id}.jpg"):
        os.remove(f"database/{person_id}.jpg")
    
    return {"status": "success", "message": f"Pessoa {person_id} removida com sucesso"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)