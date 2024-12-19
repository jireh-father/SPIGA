import os
import cv2
import argparse
import numpy as np
from pathlib import Path

# SPIGA 관련 import
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from spiga.demo.visualize.plotter import Plotter

def process_images(input_folder, output_folder, spiga_dataset='wflw', show_attributes=None):
    if show_attributes is None:
        show_attributes = ['landmarks', 'headpose']

    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 지원하는 이미지 확장자
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # SPIGA 초기화
    processor = SPIGAFramework(ModelConfig(spiga_dataset))
    plotter = Plotter()

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

            # 이미지 크기 조정 (너무 큰 이미지는 처리 속도가 느림)
            h, w = image.shape[:2]
            max_size = 1024
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
                h, w = new_h, new_w

            # 얼굴 특징점 추출
            bbox = [w//4, h//4, w//2, h//2]  # 이미지 중앙 영역을 bbox로 사용
            features = processor.inference(image, [bbox])
            
            if features and len(features['landmarks']) > 0:
                # 결과 시각화
                canvas = image.copy()
                
                if 'landmarks' in show_attributes:
                    landmarks = np.array(features['landmarks'][0])
                    # 랜드마크 좌표를 이미지 크기에 맞게 스케일 조정
                    landmarks[:, 0] *= w
                    landmarks[:, 1] *= h
                    canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
                
                if 'headpose' in show_attributes:
                    headpose = np.array(features['headpose'][0])
                    canvas = plotter.hpose.draw_headpose(canvas, 
                                                       [0, 0, w, h],  # 전체 이미지 영역 사용
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