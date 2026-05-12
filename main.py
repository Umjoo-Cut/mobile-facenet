import os
import cv2
import pickle
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import time
import math


TESTER_POSITION = (320, 300)   # 고정형 측정기 위치, 화면에서 조정 필요
DISTANCE_THRESHOLD = 80        # 입과 측정기 거리 기준
HOLD_SECONDS = 2.0             # 몇 초 유지해야 PASS인지


class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def detect_largest_face(self, frame):
        aligned_face, bbox = self.align_face(frame)

        if aligned_face is None:
            return None

        return bbox

    def crop_face(self, frame, face_box):
        aligned_face, _ = self.align_face(frame)
        return aligned_face

    def align_face(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark

        # MediaPipe FaceMesh 주요 포인트
        # 왼쪽 눈, 오른쪽 눈, 코끝, 왼쪽 입꼬리, 오른쪽 입꼬리
        left_eye = self._landmark_to_point(landmarks[33], w, h)
        right_eye = self._landmark_to_point(landmarks[263], w, h)
        nose = self._landmark_to_point(landmarks[1], w, h)
        mouth_left = self._landmark_to_point(landmarks[61], w, h)
        mouth_right = self._landmark_to_point(landmarks[291], w, h)

        src = np.array(
            [left_eye, right_eye, nose, mouth_left, mouth_right],
            dtype=np.float32
        )

        # 112x112 얼굴 정렬 기준점
        # ArcFace/MobileFaceNet 계열에서 자주 쓰는 기준점
        dst = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32
        )

        matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

        if matrix is None:
            return None, None

        aligned_face = cv2.warpAffine(frame, matrix, (112, 112))

        xs = [p[0] for p in src]
        ys = [p[1] for p in src]

        x1 = int(max(0, min(xs) - 60))
        y1 = int(max(0, min(ys) - 70))
        x2 = int(min(w, max(xs) + 60))
        y2 = int(min(h, max(ys) + 80))

        bbox = (x1, y1, x2 - x1, y2 - y1)

        return aligned_face, bbox

    def _landmark_to_point(self, landmark, width, height):
        return [
            landmark.x * width,
            landmark.y * height
        ]
    

    def get_mouth_center(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # 입 주변 주요 포인트
        mouth_indices = [13, 14, 61, 291, 78, 308]

        xs = []
        ys = []

        for idx in mouth_indices:
            xs.append(landmarks[idx].x * w)
            ys.append(landmarks[idx].y * h)

        mouth_x = int(sum(xs) / len(xs))
        mouth_y = int(sum(ys) / len(ys))

        return (mouth_x, mouth_y)

        
    def draw_face_mesh(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame

        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        return frame


class MobileFaceNetModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, face_img):
        face = cv2.resize(face_img, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5

        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        return face

    def get_embedding(self, face_img):

        if face_img is None:
            return None

        if face_img.size == 0:
            return None
        
        input_tensor = self.preprocess(face_img)
        output = self.session.run(None, {self.input_name: input_tensor})[0]

        embedding = output[0]
        embedding = embedding / np.linalg.norm(embedding)

        return embedding


class FaceDatabase:
    def __init__(self, register_dir, embedding_path, detector, model):
        self.register_dir = register_dir
        self.embedding_path = embedding_path
        self.detector = detector
        self.model = model

        os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)


    def augment_image(self, image):
        augmented_images = []

        # 원본
        augmented_images.append(image)

        # 좌우 반전
        flip = cv2.flip(image, 1)
        # augmented_images.append(flip)

        # 밝기 증가
        bright = cv2.convertScaleAbs(image, alpha=1.0, beta=20)
        augmented_images.append(bright)

        # 밝기 감소
        dark = cv2.convertScaleAbs(image, alpha=1.0, beta=-20)
        augmented_images.append(dark)

        # 대비 증가
        contrast = cv2.convertScaleAbs(image, alpha=1.15, beta=0)
        augmented_images.append(contrast)

        # 약간 블러
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        augmented_images.append(blur)

        return augmented_images

    def create_owner_embedding(self):

        if not os.path.exists(self.register_dir):
            raise FileNotFoundError(f"등록 이미지 폴더가 없습니다: {self.register_dir}")
        
        embeddings = []

        for filename in os.listdir(self.register_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(self.register_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"이미지 읽기 실패: {image_path}")
                continue

            # augmentation 적용
            # augmented_images = self.augment_image(image)
            # augmentation 미적용
            augmented_images = [image]

            for idx, aug_img in enumerate(augmented_images):

                face_box = self.detector.detect_largest_face(aug_img)

                if face_box is None:
                    print(f"얼굴 검출 실패: {image_path} aug:{idx}")
                    continue

                face_img = self.detector.crop_face(aug_img, face_box)

                try:
                    embedding = self.model.get_embedding(face_img)

                    if embedding is None:
                        print("임베딩 생성 실패")
                        continue
                    embeddings.append(embedding)

                    print(f"임베딩 생성 완료: {filename} aug:{idx}")

                except Exception as e:
                    print(f"임베딩 생성 실패: {e}")

        if len(embeddings) == 0:
            raise Exception("등록 가능한 얼굴 이미지가 없습니다.")

        # 평균 임베딩 생성
        owner_embedding = np.mean(embeddings, axis=0)

        # 정규화
        owner_embedding = owner_embedding / np.linalg.norm(owner_embedding)

        with open(self.embedding_path, "wb") as f:
            pickle.dump(owner_embedding, f)

        print(f"대표 임베딩 저장 완료")
        print(f"총 임베딩 개수: {len(embeddings)}")

    def load_owner_embedding(self):
        if os.path.exists(self.embedding_path):
            with open(self.embedding_path, "rb") as f:
                print("저장된 얼굴 임베딩 로드 완료")
                return pickle.load(f)

        if not os.path.exists(self.register_dir):
            raise FileNotFoundError(
                f"등록 이미지 폴더가 없습니다: {self.register_dir}\n"
                f"처음 실행이라면 register/owner 폴더에 얼굴 이미지를 넣어주세요.\n"
                f"이미 등록을 완료했다면 embeddings/owner_embedding.pkl 파일이 필요합니다."
            )

        self.create_owner_embedding()

        with open(self.embedding_path, "rb") as f:
            return pickle.load(f)


class FaceAuthenticator:
    def __init__(self, owner_embedding, threshold=0.55):
        self.owner_embedding = owner_embedding
        self.threshold = threshold

    def cosine_similarity(self, emb1, emb2):
        return float(np.dot(emb1, emb2))

    def authenticate(self, current_embedding):
        similarity = self.cosine_similarity(
            self.owner_embedding,
            current_embedding
        )

        is_pass = similarity >= self.threshold
        return is_pass, similarity


class FaceAuthApp:
    def __init__(self):
        self.detector = FaceDetector()
        self.model = MobileFaceNetModel("models/w600k_mbf.onnx")

        self.database = FaceDatabase(
            register_dir="register/owner",
            embedding_path="embeddings/owner_embedding.pkl",
            detector=self.detector,
            model=self.model
        )

        owner_embedding = self.database.load_owner_embedding()

        self.authenticator = FaceAuthenticator(
            owner_embedding=owner_embedding,
            threshold=0.55
        )

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        print("ESC 키를 누르면 종료됩니다.")

        near_start_time = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            face_box = self.detector.detect_largest_face(frame)

            if face_box is None:
                status = "NO FACE"
                similarity = 0.0
                color = (0, 255, 255)
            else:
                x, y, w, h = face_box
                face_img = self.detector.crop_face(frame, face_box)

                current_embedding = self.model.get_embedding(face_img)
                
                if current_embedding is None:
                    status = "NO FACE"
                    similarity = 0.0
                    color = (0, 255, 255)
                else:
                    is_pass, similarity = self.authenticator.authenticate(
                        current_embedding
                    )

                if is_pass:
                    status = "PASS"
                    color = (0, 255, 0)
                else:
                    status = "FAIL"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(
                frame,
                f"{status} sim:{similarity:.3f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3
            )

            mouth_center = self.detector.get_mouth_center(frame)

            tester_x, tester_y = TESTER_POSITION

            cv2.circle(frame, (tester_x, tester_y), 8, (255, 0, 0), -1)

            cv2.putText(
                frame,
                "Tester",
                (tester_x - 30, tester_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            mouth_near_tester = False
            hold_time = 0.0

            if mouth_center is not None:

                mouth_x, mouth_y = mouth_center

                cv2.circle(frame, (mouth_x, mouth_y), 8, (0, 255, 255), -1)

                cv2.putText(
                    frame,
                    "Mouth",
                    (mouth_x - 30, mouth_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                distance = math.sqrt(
                    (mouth_x - tester_x) ** 2 +
                    (mouth_y - tester_y) ** 2
                )

                if distance < DISTANCE_THRESHOLD:

                    mouth_near_tester = True

                    if near_start_time is None:
                        near_start_time = time.time()

                    hold_time = time.time() - near_start_time

                else:
                    near_start_time = None

                cv2.line(
                    frame,
                    (mouth_x, mouth_y),
                    (tester_x, tester_y),
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Distance: {int(distance)}",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

            else:
                near_start_time = None

            if status == "PASS" and mouth_near_tester and hold_time >= HOLD_SECONDS:

                final_status = "FINAL PASS"
                final_color = (0, 255, 0)

            else:

                final_status = "FINAL FAIL"
                final_color = (0, 0, 255)

            cv2.putText(
                frame,
                f"{final_status} hold:{hold_time:.1f}s",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                final_color,
                3
            )


            frame = self.detector.draw_face_mesh(frame)
            cv2.imshow("MobileFaceNet Face Auth", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = FaceAuthApp()
    app.run()