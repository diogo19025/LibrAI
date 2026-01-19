import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")


def count_fingers(hand_landmarks, handedness_label=None):
    """
    Conta dedos levantados usando landmarks do MediaPipe.
    hand_landmarks: lista de 21 landmarks (cada lm tem .x e .y normalizados 0..1)
    handedness_label: "Left" ou "Right" (se disponível)
    """
    # IDs padrão do MediaPipe
    # Polegar: TIP=4, IP=3
    # Indicador: TIP=8, PIP=6
    # Médio: TIP=12, PIP=10
    # Anelar: TIP=16, PIP=14
    # Mindinho: TIP=20, PIP=18
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    fingers_up = 0

    # 1) Polegar (melhor por X)
    thumb_tip = hand_landmarks[tips[0]]
    thumb_ip = hand_landmarks[pips[0]]

    # Como você usa flip(frame, 1), a lógica pode precisar de inversão dependendo do handedness.
    if handedness_label == "Right":
        fingers_up += 1 if thumb_tip.x < thumb_ip.x else 0
    elif handedness_label == "Left":
        fingers_up += 1 if thumb_tip.x > thumb_ip.x else 0
    else:
        # fallback sem handedness
        fingers_up += 1 if abs(thumb_tip.x - thumb_ip.x) > 0.03 else 0

    # 2) Outros 4 dedos (por Y: ponta acima da junta => levantado)
    for tip_id, pip_id in zip(tips[1:], pips[1:]):
        if hand_landmarks[tip_id].y < hand_landmarks[pip_id].y:
            fingers_up += 1

    return fingers_up


def main():
    print("CWD:", os.getcwd())
    print("MODEL_PATH:", MODEL_PATH)
    print("MODEL_PATH exists?", os.path.exists(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Não achei o modelo '{MODEL_PATH}'. Coloque ele na mesma pasta do script "
            f"ou passe o caminho completo."
        )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera opened?", cap.isOpened())
    if not cap.isOpened():
        raise RuntimeError("Não consegui abrir a câmera.")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2
    )

    print("Criando HandLandmarker...")
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        print("HandLandmarker criado. Entrando no loop da camera...")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler frame da câmera.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    # Handedness (Left/Right) se existir
                    handedness_label = None
                    if hasattr(result, "handedness") and result.handedness:
                        # geralmente vem como lista de listas; pegamos a top categoria
                        raw_label = result.handedness[i][0].category_name  # Left / Right

                        # INVERTE propositalmente
                        if raw_label == "Left":
                            handedness_label = "Right"
                        elif raw_label == "Right":
                            handedness_label = "Left"
                        else:
                            handedness_label = raw_label

                    # Desenha landmarks
                    for lm in hand_landmarks:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    # Conta dedos
                    fingers = count_fingers(hand_landmarks, handedness_label)

                    # Texto na tela (um por mão)
                    cv2.putText(
                        frame,
                        f"Hand {i+1} ({handedness_label or 'Unknown'}): {fingers} fingers",
                        (10, 40 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2
                    )

            cv2.imshow("Hand Landmarks (MediaPipe Tasks)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
