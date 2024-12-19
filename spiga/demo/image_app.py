import os
import cv2
import argparse
import numpy as np
from pathlib import Path

# SPIGA 관련 import
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from spiga.demo.visualize.plotter import Plotter
from spiga.demo.analyze.track.retinasort.face_tracker import RetinaFace

def process_images(input_folder, output_folder, spiga_dataset='wflw', show_attributes=None):
    if show_attributes is None:
        show_attributes = ['landmarks', 'headpose']

    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 지원하는 이미지 확장자
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # SPIGA와 얼굴 검출기 초기화
    processor = SPIGAFramework(ModelConfig(spiga_dataset))
    plotter = Plotter()
    face_detector = RetinaFace()

    # 입력 폴더의 모든 이미지 파일 처리
    for filename in os.listdir(input_folder):
        if Path(filename).suffix.lower() in image_extensions:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f'spiga_{filename}')

            # 이미지 읽기
            image = cv2.imread(input_path)
            if image is None:
                print(f"이미지를 읽을 수 없습니다: {filename}")
                continue

            # 얼굴 검출
            h, w = image.shape[:2]
            face_detector.set_input_shape(h, w)
            detections = face_detector.detect_faces(image)

            if detections:
                # 결과 시각화
                canvas = image.copy()
                
                for det in detections:
                    bbox = [det[0], det[1], det[2]-det[0], det[3]-det[1]]  # x,y,w,h 형식으로 변환
                    features = processor.inference(image, [bbox])

                    if features and len(features['landmarks']) > 0:
                        if 'landmarks' in show_attributes:
                            landmarks = np.array(features['landmarks'][0])
                            canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
                        
                        if 'headpose' in show_attributes:
                            headpose = np.array(features['headpose'][0])
                            canvas = plotter.hpose.draw_headpose(canvas, 
                                                               [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], 
                                                               headpose[:3], 
                                                               headpose[3:], 
                                                               euler=True)

                # 결과 저장
                cv2.imwrite(output_path, canvas)
                print(f"처리 완료: {filename}")
            else:
                print(f"얼굴을 찾을 수 없습니다: {filename}")

def main():
    parser = argparse.ArgumentParser(description='SPIGA 얼굴 특징점 일괄 처리 프로그램')
    parser.add_argument('-i', '--input', 
                        type=str, 
                        required=True,
                        help='입력 이미지가 있는 폴더 경로')
    parser.add_argument('-o', '--output', 
                        type=str, 
                        required=True,
                        help='결과 이미지를 저장할 폴더 경로')
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
                        help='표시할 얼굴 특징 선택')

    args = parser.parse_args()
    
    process_images(args.input, 
                  args.output, 
                  spiga_dataset=args.dataset,
                  show_attributes=args.show)

if __name__ == "__main__":
    main() 