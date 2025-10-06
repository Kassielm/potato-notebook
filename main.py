import cv2
import logging
import os
from datetime import datetime
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisionSystem:
    def __init__(self, model_path: str = 'pepsico-v1-n.onnx', camera_index: int = 0, conf_threshold: float = 0.35):
        self.model = YOLO(model_path)
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.save_path = 'c:/potato-identifier/'

        self.colors = {
            'OK': (0, 255, 0),
            'PODRE': (255, 0, 0),
            'PEDRA': (0, 0, 255),
            'PEDRA-NA-BATATA': (0, 0, 255),
            'BATATA-COM-PEDRA': (0, 0, 255),
        }
        self.window_name = 'Sistema de Visao Conecsa'
        self.resolution = (1980, 1080)
        self.camera = None

        self.saved_stone_ids = set()

    def init_camera(self) -> bool:
        """Inicializa a webcam"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                logger.error(f"Erro ao abrir webcam no índice {self.camera_index}")
                return False
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            logging.info("Webcam encontrada e aberta com sucesso.")
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            return True
        except Exception as e:
            logger.error(f"Erro ao inicializar webcam: {e}")
            logger.error("Nenhuma webcam encontrada. Verifique a conexão.")
            return False

    def save_stone_image(self, frame) -> None:
        """Salva a imagem quando uma pedra é detectada."""
        try:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                logger.info(f"Diretório criado: '{self.save_path}'")

            # Cria um nome de arquivo único com data e hora
            timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S-") + f"{datetime.now().microsecond // 1000:03d}"
            filename = os.path.join(self.save_path, f'{timestamp}.jpg')
            
            # Salva o frame inteiro
            cv2.imwrite(filename, frame)
            logger.info(f"Print da tela salvo em: {filename}")

        except Exception as e:
            logger.error(f"Falha ao salvar a imagem da pedra: {e}")

    def process_frame(self) -> None:
        """Processa o frame com YOLO, rastreia objetos e exibe os resultados."""
        while True:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning('Falha ao capturar frame da webcam')
                    continue

                original_h, original_w = frame.shape[:2]
                small_frame = cv2.resize(frame, (640, 640))
                
                results = self.model.track(small_frame, conf=self.conf_threshold, persist=True)

                scale_x = original_w / 640
                scale_y = original_h / 640

                frame_com_desenhos = frame.copy()

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy
                    classes = results[0].boxes.cls
                    scores = results[0].boxes.conf
                    track_ids = results[0].boxes.id.int().tolist()

                    for box, cls, score, track_id in zip(boxes, classes, scores, track_ids):
                        x1, y1, x2, y2 = int(box[0] * scale_x), int(box[1] * scale_y), int(box[2] * scale_x), int(box[3] * scale_y)
                        
                        label = self.model.names[int(cls)]
                        color = self.colors.get(label, (0, 255, 0))
                        
                        display_text = f'{label}: {score:.2f}'
                        cv2.rectangle(frame_com_desenhos, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_com_desenhos, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        if (label == 'PEDRA' or label == 'PEDRA-NA-BATATA' or label == 'BATATA-COM-PEDRA') and track_id not in self.saved_stone_ids:
                            self.save_stone_image(frame_com_desenhos)
                            self.saved_stone_ids.add(track_id)

                cv2.imshow(self.window_name, frame_com_desenhos)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                logger.error(f"Erro ao processar o frame: {e}")
                continue

    def cleanup(self) -> None:
        """Libera a câmera e fecha as janelas do OpenCV."""
        try:
            if self.camera and self.camera.isOpened():
                self.camera.release()
            cv2.destroyAllWindows()
            logger.info("Recursos liberados com sucesso")
        except Exception as e:
            logger.error(f"Erro durante a limpeza: {e}")

    def __enter__(self):
        self.init_camera()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

def main():
    """Função principal para rodar o sistema de visão."""
    with VisionSystem(model_path='pedra-podre.onnx', camera_index=1, conf_threshold=0.45) as vision_system:
        if vision_system.camera and vision_system.camera.isOpened():
            vision_system.process_frame()
        else:
            print("Saindo do programa pois a webcam não pôde ser inicializada.")

if __name__ == "__main__":
    main()