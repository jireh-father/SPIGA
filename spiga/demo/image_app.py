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

class ModelCache:
    def __init__(self):
        self.spiga_processors = {}  # dataset별 SPIGA 모델 캐시
        self.mediapipe_model = None
        self.mediapipe_detector = None  # face detection용 MediaPipe 모델
        self.dlib_detector = None
        self.dlib_predictor = None
        self.face_detector = None
        
    def get_spiga(self, dataset):
        if dataset not in self.spiga_processors:
            self.spiga_processors[dataset] = SPIGAFramework(ModelConfig(dataset))
        return self.spiga_processors[dataset]
    
    def get_mediapipe(self):
        if self.mediapipe_model is None:
            self.mediapipe_model = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        return self.mediapipe_model
    
    def get_mediapipe_detector(self):
        if self.mediapipe_detector is None:
            self.mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 0=근거리, 1=원거리
                min_detection_confidence=0.5
            )
        return self.mediapipe_detector
    
    def get_dlib(self):
        if self.dlib_detector is None:
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        return self.dlib_detector, self.dlib_predictor
    
    def get_face_detector(self):
        if self.face_detector is None:
            self.face_detector = dlib.get_frontal_face_detector()
        return self.face_detector
    
    def clear_model(self, model_name):
        if model_name == 'spiga':
            self.spiga_processors.clear()
        elif model_name == 'mediapipe':
            if self.mediapipe_model:
                self.mediapipe_model.close()
            if self.mediapipe_detector:
                self.mediapipe_detector.close()
            self.mediapipe_model = None
            self.mediapipe_detector = None
        elif model_name == 'dlib':
            self.dlib_detector = None
            self.dlib_predictor = None

def get_all_image_files(input_folder):
    """재귀적으로 모든 이미지 파일 찾기"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return image_files

def pad_and_resize(image, target_size=512):
    """이미지를 정사각형으로 패딩한 후 target_size로 리사이즈"""
    h, w = image.shape[:2]
    
    # 정사각형 만들기 위한 패딩 계산
    max_side = max(h, w)
    pad_h = (max_side - h) // 2
    pad_w = (max_side - w) // 2
    
    # 패딩 추가 (위, 아래, 좌, 우)
    padded = cv2.copyMakeBorder(
        image,
        pad_h, pad_h + (max_side - h) % 2,  # 홀수인 경우 아래쪽에 1픽셀 더 추가
        pad_w, pad_w + (max_side - w) % 2,  # 홀수인 경우 오른쪽에 1픽셀 더 추가
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # 검정색으로 패딩
    )
    
    # target_size로 리사이즈
    resized = cv2.resize(padded, (target_size, target_size))
    return resized

def detect_and_crop_face(image, model_cache, margin_ratio=0.3):
    """MediaPipe로 얼굴 검출 후 크롭"""
    detector = model_cache.get_mediapipe_detector()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)
    
    if results.detections:
        # 첫 번째 얼굴만 사용
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # 상대 좌표를 절대 좌표로 변환
        ih, iw = image.shape[:2]
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)
        
        # 여유 공간 추가 (margin_ratio * 100%)
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)
        
        # 이미지 경계 확인
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(iw, x + w + margin_x)
        y2 = min(ih, y + h + margin_y)
        
        face_crop = image[y1:y2, x1:x2]
        # 크롭된 얼굴을 정사각형으로 패딩하고 512x512로 리사이즈
        face_crop = pad_and_resize(face_crop, target_size=512)
        return face_crop, (x1, y1, x2, y2)
    return None, None

def process_spiga(image, model_cache, spiga_dataset='wflw', show_attributes=None, margin_ratio=0.3):
    """SPIGA 모델로 랜드마크 추출"""
    if show_attributes is None:
        show_attributes = ['landmarks', 'headpose']
    
    # 얼굴 검출 및 크롭    
    face_crop, crop_coords = detect_and_crop_face(image, model_cache, margin_ratio)
    if face_crop is None:
        return None
        
    h, w = face_crop.shape[:2]
    x0, y0 = w//4, h//4
    bbox_w, bbox_h = w//2, h//2
    bbox = [x0, y0, bbox_w, bbox_h]
    
    processor = model_cache.get_spiga(spiga_dataset)
    plotter = Plotter()
    
    features = processor.inference(face_crop, [bbox])
    if features and len(features['landmarks']) > 0:
        canvas = copy.deepcopy(face_crop)
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

def process_mediapipe(image, model_cache, margin_ratio=0.3):
    """MediaPipe로 랜드마크 추출"""
    # 얼굴 검출 및 크롭
    face_crop, crop_coords = detect_and_crop_face(image, model_cache, margin_ratio)
    if face_crop is None:
        return None
        
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    face_mesh = model_cache.get_mediapipe()
    image_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        canvas = copy.deepcopy(face_crop)
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        return canvas
    return None

def process_dlib(image, model_cache, margin_ratio=0.3):
    """dlib으로 랜드마크 추출"""
    # 얼굴 검출 및 크롭
    face_crop, crop_coords = detect_and_crop_face(image, model_cache, margin_ratio)
    if face_crop is None:
        return None
        
    detector, predictor = model_cache.get_dlib()
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        canvas = copy.deepcopy(face_crop)
        for face in faces:
            landmarks = predictor(gray, face)
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
        return canvas
    return None

def process_images(input_folder, output_folder, model='spiga', spiga_dataset='wflw', show_attributes=None, margin_ratio=0.3):
    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 모든 이미지 파일 찾기
    image_files = get_all_image_files(input_folder)
    
    # 실행할 모델과 데이터셋 리스트 설정
    models_to_run = ['spiga', 'mediapipe', 'dlib'] if model == 'all' else [model]
    datasets_to_run = ['wflw', '300wpublic', '300wprivate', 'merlrav'] if spiga_dataset == 'all' else [spiga_dataset]
    
    # 모델 캐시 초기화
    model_cache = ModelCache()
    
    # 모델별로 처리 (all 옵션일 때 메모리 효율을 위해)
    for current_model in models_to_run:
        print(f"\n=== {current_model.upper()} 모델 처리 시작 ===")
        
        for image_path in image_files:
            # 상대 경로 유지를 위한 처리
            rel_path = os.path.relpath(image_path, input_folder)
            output_subdir = os.path.dirname(os.path.join(output_folder, rel_path))
            os.makedirs(output_subdir, exist_ok=True)
            
            # 이미지 읽기
            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지를 읽을 수 없습니다: {image_path}")
                continue

            # 얼굴 검출 및 크롭
            face_crop, crop_coords = detect_and_crop_face(image, model_cache, margin_ratio)
            if face_crop is None:
                print(f"얼굴을 찾을 수 없습니다: {rel_path}")
                continue

            # 원본 크롭 이미지 저장 (첫 번째 모델에서만 수행)
            if current_model == models_to_run[0]:
                name, ext = os.path.splitext(os.path.basename(image_path))
                orig_output_path = os.path.join(output_subdir, f'{name}_crop{ext}')
                if not os.path.exists(orig_output_path):
                    cv2.imwrite(orig_output_path, face_crop)
                    print(f"크롭된 원본 이미지 저장: {rel_path}")

            if current_model == 'spiga':
                # SPIGA 모델의 경우 각 데이터셋별로 처리
                for current_dataset in datasets_to_run:
                    name, ext = os.path.splitext(os.path.basename(image_path))
                    output_path = os.path.join(output_subdir, f'{name}_spiga_{current_dataset}{ext}')
                    
                    if os.path.exists(output_path):
                        print(f"이미 처리됨 (spiga-{current_dataset}): {rel_path}")
                        continue
                    
                    result = process_spiga(image, model_cache, current_dataset, show_attributes, margin_ratio)
                    if result is not None:
                        cv2.imwrite(output_path, result)
                        print(f"처리 완료 (spiga-{current_dataset}): {rel_path}")
                    else:
                        print(f"얼굴을 찾을 수 없습니다 (spiga-{current_dataset}): {rel_path}")
            else:
                name, ext = os.path.splitext(os.path.basename(image_path))
                output_path = os.path.join(output_subdir, f'{name}_{current_model}{ext}')
                
                if os.path.exists(output_path):
                    print(f"이미 처리됨 ({current_model}): {rel_path}")
                    continue
                
                result = None
                if current_model == 'mediapipe':
                    result = process_mediapipe(image, model_cache, margin_ratio)
                elif current_model == 'dlib':
                    result = process_dlib(image, model_cache, margin_ratio)

                if result is not None:
                    cv2.imwrite(output_path, result)
                    print(f"처리 완료 ({current_model}): {rel_path}")
                else:
                    print(f"얼굴을 찾을 수 없습니다 ({current_model}): {rel_path}")
        
        # 현재 모델 처리가 끝나면 메모리에서 해제
        model_cache.clear_model(current_model)
        print(f"=== {current_model.upper()} 모델 처리 완료 ===\n")

def main():
    # shape predictor 모델 다운로드
    download_shape_predictor()
    
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
                        choices=['spiga', 'mediapipe', 'dlib', 'all'],
                        help='사용할 랜드마크 추출 모델')
    parser.add_argument('-d', '--dataset', 
                        type=str, 
                        default='wflw',
                        choices=['wflw', '300wpublic', '300wprivate', 'merlrav', 'all'],
                        help='SPIGA 사전 학습 가중치 데이터셋')
    parser.add_argument('-sh', '--show', 
                        nargs='+', 
                        type=str, 
                        default=['landmarks', 'headpose'],
                        choices=['bbox', 'landmarks', 'headpose'],
                        help='SPIGA 모델의 표시할 얼굴 특징 선택')
    parser.add_argument('-mr', '--margin-ratio',
                        type=float,
                        default=0.3,
                        help='얼굴 크롭 시 추가할 여유 공간 비율 (기본값: 0.3 = 30%%)')

    args = parser.parse_args()
    
    process_images(args.input, 
                  args.output,
                  model=args.model,
                  spiga_dataset=args.dataset,
                  show_attributes=args.show,
                  margin_ratio=args.margin_ratio)

def download_shape_predictor():
    """shape_predictor_68_face_landmarks.dat 파일 다운로드"""
    import bz2
    import urllib.request
    import os
    
    dat_file = "shape_predictor_68_face_landmarks.dat"
    bz2_file = dat_file + ".bz2"
    
    # 이미 파일이 존재하는지 확인
    if os.path.exists(dat_file):
        print(f"{dat_file} 파일이 이미 존재합니다.")
        return
        
    print("shape predictor 모델 다운로드 중...")
    url = f"http://dlib.net/files/{bz2_file}"
    
    try:
        # bz2 파일 다운로드
        urllib.request.urlretrieve(url, bz2_file)
        
        # bz2 압축 해제
        with bz2.BZ2File(bz2_file) as fr, open(dat_file, 'wb') as fw:
            fw.write(fr.read())
            
        # bz2 파일 삭제
        os.remove(bz2_file)
        print("다운로드 완료!")
        
    except Exception as e:
        print(f"다운로드 실패: {str(e)}")
        if os.path.exists(bz2_file):
            os.remove(bz2_file)

if __name__ == "__main__":
    main() 