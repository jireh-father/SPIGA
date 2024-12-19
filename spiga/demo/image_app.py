import os
import cv2
import copy
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

            # 이미지 크기 가져오기
            h, w = image.shape[:2]
            
            # bbox 설정 (이미지 중앙 영역)
            x0, y0 = w//4, h//4
            bbox_w, bbox_h = w//2, h//2
            bbox = [x0, y0, bbox_w, bbox_h]

            # 얼굴 특징점 추출
            features = processor.inference(image, [bbox])
            
            if features and len(features['landmarks']) > 0:
                # 결과 시각화
                canvas = copy.deepcopy(image)
                landmarks = np.array(features['landmarks'][0])
                headpose = np.array(features['headpose'][0])

                # Plot features
                if 'landmarks' in show_attributes:
                    canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
                
                if 'headpose' in show_attributes:
                    canvas = plotter.hpose.draw_headpose(canvas, 
                                                       [x0, y0, x0+bbox_w, y0+bbox_h],
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