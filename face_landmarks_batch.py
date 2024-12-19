import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

def process_images(input_folder, output_folder):
    # MediaPipe Face Mesh 초기화
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 지원하는 이미지 확장자
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        # 입력 폴더의 모든 이미지 파일 처리
        for filename in os.listdir(input_folder):
            if Path(filename).suffix.lower() in image_extensions:
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, f'landmark_{filename}')

                # 이미지 읽기
                image = cv2.imread(input_path)
                if image is None:
                    print(f"이미지를 읽을 수 없습니다: {filename}")
                    continue

                # BGR을 RGB로 변환
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    # 랜드마크 그리기
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                    # 결과 이미지 저장
                    cv2.imwrite(output_path, image)
                    print(f"처리 완료: {filename}")
                else:
                    print(f"얼굴을 찾을 수 없습니다: {filename}")

if __name__ == "__main__":
    # 입력 폴더와 출력 폴더 지정
    input_folder = "input_images"  # 입력 이미지가 있는 폴더
    output_folder = "output_images"  # 결과를 저장할 폴더
    
    process_images(input_folder, output_folder) 