import os
import cv2
import argparse
import pkg_resources
from pathlib import Path

# SPIGA 관련 import
import spiga.demo.analyze.track.get_tracker as tr
import spiga.demo.analyze.extract.spiga_processor as pr_spiga
from spiga.demo.analyze.analyzer import VideoAnalyzer
from spiga.demo.visualize.viewer import Viewer

def process_images(input_folder, output_folder, spiga_dataset='wflw', tracker='RetinaSort', show_attributes=None):
    if show_attributes is None:
        show_attributes = ['landmarks', 'headpose']

    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 지원하는 이미지 확장자
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # SPIGA 초기화
    faces_tracker = tr.get_tracker(tracker)
    processor = pr_spiga.SPIGAProcessor(dataset=spiga_dataset)
    faces_analyzer = VideoAnalyzer(faces_tracker, processor=processor)

    # Viewer 초기화 (이미지 저장용)
    viewer = Viewer('image_app', width=None, height=None)

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

            # 이미지 크기 설정
            faces_tracker.detector.set_input_shape(image.shape[0], image.shape[1])

            # 이미지 처리
            faces_analyzer.process_frame(image)

            # 결과 시각화 및 저장
            viewer.process_image(image, 
                               drawers=[faces_analyzer], 
                               show_attributes=show_attributes,
                               save_path=output_path,
                               show=False)
            
            print(f"처리 완료: {filename}")

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
    parser.add_argument('-t', '--tracker', 
                        type=str, 
                        default='RetinaSort',
                        choices=['RetinaSort', 'RetinaSort_Res50'], 
                        help='트래커 이름')
    parser.add_argument('-sh', '--show', 
                        nargs='+', 
                        type=str, 
                        default=['fps', 'face_id', 'landmarks', 'headpose'],
                        choices=['fps', 'bbox', 'face_id', 'landmarks', 'headpose'],
                        help='표시할 얼굴 특징 선택')

    args = parser.parse_args()
    
    process_images(args.input, 
                  args.output, 
                  spiga_dataset=args.dataset,
                  tracker=args.tracker,
                  show_attributes=args.show)

if __name__ == "__main__":
    main() 