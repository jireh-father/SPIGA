import os
import cv2
import copy
import dlib
import numpy as np
import mediapipe as mp
from pathlib import Path
import argparse

# SPIGA 관련 import
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from spiga.demo.visualize.plotter import Plotter

def get_all_image_files(input_folder):
    """재귀적으로 모든 이미지 파일 찾기"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return image_files

def process_spiga(image, spiga_dataset='wflw', show_attributes=None):
    """SPIGA 모델로 랜드마크 추출"""
    if show_attributes is None:
        show_attributes = ['landmarks', 'headpose']
        
    h, w = image.shape[:2]
    x0, y0 = w//4, h//4
    bbox_w, bbox_h = w//2, h//2
    bbox = [x0, y0, bbox_w, bbox_h]
    
    processor = SPIGAFramework(ModelConfig(spiga_dataset))
    plotter = Plotter()
    
    features = processor.inference(image, [bbox])
    if features and len(features['landmarks']) > 0:
        canvas = copy.deepcopy(image)
        landmarks = np.array(features['landmarks'][0])
        headpose = np.array(features['headpose'][0])

        if 'landmarks' in show_attributes:
            canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
        if 'headpose' in show_attributes:
            canvas = plotter.hpose.draw_headpose(canvas, 
                                               [x0, y0, x0+bbox_w, y0+bbox_h],
                                               headpose[:3], 
                                               headpose[3:], 
                                               euler=True)
        return canvas
    return None

def process_mediapipe(image):
    """MediaPipe로 랜드마크 추출"""
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            canvas = copy.deepcopy(image)
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=canvas,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=canvas,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            return canvas
    return None

def process_dlib(image):
    """dlib으로 랜드마크 추출"""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        canvas = copy.deepcopy(image)
        for face in faces:
            landmarks = predictor(gray, face)
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
        return canvas
    return None

def process_images(input_folder, output_folder, model='spiga', spiga_dataset='wflw', show_attributes=None):
    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 모든 이미지 파일 찾기
    image_files = get_all_image_files(input_folder)
    
    for image_path in image_files:
        # 상대 경로 유지를 위한 처리
        rel_path = os.path.relpath(image_path, input_folder)
        output_subdir = os.path.dirname(os.path.join(output_folder, rel_path))
        os.makedirs(output_subdir, exist_ok=True)
        
        # 출력 파일명 생성
        name, ext = os.path.splitext(os.path.basename(image_path))
        if model == 'spiga':
            output_path = os.path.join(output_subdir, f'spiga_{name}_{spiga_dataset}{ext}')
        else:
            output_path = os.path.join(output_subdir, f'{model}_{name}{ext}')

        # 이미지 처리
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            continue

        # 모델별 처리
        result = None
        if model == 'spiga':
            result = process_spiga(image, spiga_dataset, show_attributes)
        elif model == 'mediapipe':
            result = process_mediapipe(image)
        elif model == 'dlib':
            result = process_dlib(image)

        if result is not None:
            cv2.imwrite(output_path, result)
            print(f"처리 완료: {rel_path}")
        else:
            print(f"얼굴을 찾을 수 없습니다: {rel_path}")

def main():
    parser = argparse.ArgumentParser(description='얼굴 랜드마크 일괄 처리 프로그램')
    parser.add_argument('-i', '--input', 
                        type=str, 
                        required=True,
                        help='입력 이미지가 있는 폴더 경로')
    parser.add_argument('-o', '--output', 
                        type=str, 
                        required=True,
                        help='결과 이미지를 저장할 폴더 경로')
    parser.add_argument('-m', '--model',
                        type=str,
                        default='spiga',
                        choices=['spiga', 'mediapipe', 'dlib'],
                        help='사용할 랜드마크 추출 모델')
    parser.add_argument('-d', '--dataset', 
                        type=str, 
                        default='wflw',
                        choices=['wflw', '300wpublic', '300wprivate', 'merlrav'],
                        help='SPIGA 사전 학습 가중치 데이터셋')
    parser.add_argument('-sh', '--show', 
                        nargs='+', 
                        type=str, 
                        default=['landmarks', 'headpose'],
                        choices=['bbox', 'landmarks', 'headpose'],
                        help='SPIGA 모델의 표시할 얼굴 특징 선택')

    args = parser.parse_args()
    
    process_images(args.input, 
                  args.output,
                  model=args.model,
                  spiga_dataset=args.dataset,
                  show_attributes=args.show)

if __name__ == "__main__":
    main() 