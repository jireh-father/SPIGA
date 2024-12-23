import os
import cv2
import copy
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import argparse
from pathlib import Path

# SPIGA 관련 import
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from spiga.demo.visualize.plotter import Plotter

# 얼굴 특징 계층 구조 정의
FEATURE_HIERARCHY = {
    'mendible': {  # 하악 턱끝부터 귀아래까지의 뼈
        # 턱부터 수정해야함
        'gonion_width': ['widest', 'wide', 'normal', 'narrow'],  # 상악 턱 너비
        'gonion_height': ['normal', 'long', 'short'],  # 상악 턱 높이
        'mandibular_angle': ['angular', 'smooth'],  # 상악 턱 각도
        'mental_width': ['wide', 'normal', 'narrow'],  # 턱끝, gnation(mental protuberance의 가장 아래부분) + mental tubercle
        'mandible_mid_angle': ['angular', 'smooth'],  # 아래턱 각도
        'mandible_height': ['normal', 'long', 'short'],  # 아래턱 높이
        'mandible_side': ['normal', 'prognathism', 'retrognathia']  # 아래턱 측면
    },
    'forehead': {  # 이마
        'length': ['normal', 'long', 'short'],  # 길이
        'top_width': ['normal', 'narrow', 'wide', 'very_wide'],  # 상단
        'bottom_width': ['normal', 'wide', 'narrow'],  # 하단
        'corner_angle': ['angular', 'smooth'],  # 각도
        'hairline': ['normal', 'm_shaped']  # M자 여부
    },
    'face': {  # 얼굴
        'ratio': ['long', 'short', 'moderate', 'square'],  # 가로세로 비율
        'upper_lower': ['long', 'equal', 'short']  # 상하 비율
    },
    'cheekbone': {  # 광대
        'prominence': ['prominent', 'normal']  # 돌출 여부
    }
}

class FeatureValue(Enum):
    """모든 특징값을 포함하는 Enum"""
    # 너비 관련
    NONE = auto()
    WIDE = auto()
    NARROW = auto()
    WIDEST = auto()
    NORMAL = auto()
    VERY_WIDE = auto()
    
    # 높이/길이 관련
    LONG = auto()
    SHORT = auto()
    
    # 각도 관련
    ANGULAR = auto()
    SMOOTH = auto()
    
    # 측면
    PROGNATHISM = auto()
    RETROGNATHIA = auto()
    
    # 모양
    M_SHAPED = auto()
    
    # 비율
    MODERATE = auto()
    SQUARE = auto()
    EQUAL = auto()
    
    # 돌출
    PROMINENT = auto()

@dataclass
class FeatureAnalysis:
    """특징 분석 결과를 저장하는 데이터 클래스"""
    feature_path: str  # 특징의 전체 경로 (예: "chin/bottom/width")
    value: FeatureValue  # 분석된 값
    confidence: float  # 신뢰도 (0~1)
    landmarks: List[int]  # 사용된 랜드마크 인덱스들

class FaceShapeAnalyzer:
    """얼굴형 분석기 클래스"""
    
    def __init__(self, dataset: str = 'wflw'):
        self.spiga = SPIGAFramework(ModelConfig(dataset))
        self.plotter = Plotter()
        
        # 랜드마크 인덱스 매핑
        self.LANDMARK_GROUPS = {
            'chin': {
                'bottom': {
                    'center': 8,  # 턱 끝점
                    'left': 6,    # 턱 왼쪽 끝점
                    'right': 10   # 턱 오른쪽 끝점
                },
                'jawline': {
                    'left': 4,    # 턱선 왼쪽
                    'right': 12   # 턱선 오른쪽
                }
            },
            'cheekbone': {
                'left': 2,        # 광대 왼쪽
                'right': 14       # 광대 오른쪽
            },
            'forehead': {
                'top': 20,        # 이마 최상단
                'left': 18,       # 이마 왼쪽
                'right': 22       # 이마 오른쪽
            }
        }
        
        # 특징별 임계값 설정
        self.THRESHOLDS = {
            'chin/width': {
                'widest': 1.2,  # 대영너비/하관너비 비율
                'wide': 1.1,
                'normal': 0.9,
                'narrow': 0.8
            },
            'chin/bottom/angle': 45,  # 각도 (도)
            'chin/bottom/height': {
                'normal': 0.31,  # 입술밑-턱끝 길이가 하관-턱끝 길이의 31%
                'long': 0.35,
                'short': 0.27
            },
            'forehead/length': {
                'normal': 0.33,  # 얼굴 길이의 1/3
                'long': 0.36,
                'short': 0.30
            },
            'face/ratio': {
                'long': 1.5,    # 얼굴 길이/너비 비율
                'moderate': 1.3,
                'short': 1.1
            }
        }
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """SPIGA 모델을 사용하여 랜드마크 추출"""
        h, w = image.shape[:2]
        x0, y0 = w//4, h//4
        bbox_w, bbox_h = w//2, h//2
        bbox = [x0, y0, bbox_w, bbox_h]
        
        features = self.spiga.inference(image, [bbox])
        if features and len(features['landmarks']) > 0:
            return np.array(features['landmarks'][0])
        return None
    
    def analyze_all_features(self, landmarks: np.ndarray) -> Dict[str, FeatureAnalysis]:
        """모든 특징을 분석하고 결과를 딕셔너리로 반환"""
        results = {}
        
        # 턱 관련 특징 분석
        results['chin/bottom/width'] = self._analyze_chin_bottom_width(landmarks)
        results['chin/bottom/height'] = self._analyze_chin_bottom_height(landmarks)
        results['chin/bottom/angle'] = self._analyze_chin_bottom_angle(landmarks)
        results['chin/width'] = self._analyze_chin_width(landmarks)
        results['chin/angle'] = self._analyze_chin_angle(landmarks)
        results['chin/height'] = self._analyze_chin_height(landmarks)
        results['chin/profile'] = self._analyze_chin_profile(landmarks)
        
        # 이마 관련 특징 분석
        results['forehead/length'] = self._analyze_forehead_length(landmarks)
        results['forehead/width/top'] = self._analyze_forehead_top_width(landmarks)
        results['forehead/width/bottom'] = self._analyze_forehead_bottom_width(landmarks)
        results['forehead/angle'] = self._analyze_forehead_angle(landmarks)
        results['forehead/shape'] = self._analyze_forehead_shape(landmarks)
        
        # 얼굴 비율 분석
        results['face/ratio'] = self._analyze_face_ratio(landmarks)
        results['face/upper_lower'] = self._analyze_face_upper_lower_ratio(landmarks)
        
        # 광대 분석
        results['cheekbone/prominence'] = self._analyze_cheekbone_prominence(landmarks)
        
        return {k: v for k, v in results.items() if v is not None}
    
    def _analyze_chin_bottom_height(self, landmarks: np.ndarray) -> FeatureAnalysis:
        """아래턱 높이 분석"""
        chin = self.LANDMARK_GROUPS['chin']['bottom']
        
        # 입술밑-턱끝 길이 계산 (임시로 이마-턱끝 길이의 비율로 계산)
        chin_height = np.linalg.norm(
            landmarks[chin['center']] - landmarks[self.LANDMARK_GROUPS['forehead']['top']]
        )
        
        # 전체 얼굴 길이 대비 비율 계산
        face_height = np.linalg.norm(
            landmarks[self.LANDMARK_GROUPS['forehead']['top']] - 
            landmarks[chin['center']]
        )
        
        ratio = chin_height / face_height
        thresholds = self.THRESHOLDS['chin/bottom/height']
        
        if ratio >= thresholds['long']:
            value = FeatureValue.LONG
            confidence = min((ratio - thresholds['long']) / 0.1 + 0.8, 1.0)
        elif ratio <= thresholds['short']:
            value = FeatureValue.SHORT
            confidence = min((thresholds['short'] - ratio) / 0.1 + 0.8, 1.0)
        else:
            value = FeatureValue.NORMAL
            confidence = 1.0 - abs(ratio - thresholds['normal']) / 0.1
        
        return FeatureAnalysis(
            feature_path='chin/bottom/height',
            value=value,
            confidence=confidence,
            landmarks=[chin['center'], self.LANDMARK_GROUPS['forehead']['top']]
        )
    
    def _analyze_chin_bottom_angle(self, landmarks: np.ndarray) -> FeatureAnalysis:
        """아래턱 각도 분석"""
        chin = self.LANDMARK_GROUPS['chin']['bottom']
        
        # 턱 끝점과 턱선 양 끝점으로 각도 계산
        center = landmarks[chin['center']]
        left = landmarks[chin['left']]
        right = landmarks[chin['right']]
        
        # 두 벡터 계산
        vector1 = left - center
        vector2 = right - center
        
        # 각도 계산 (라디안)
        angle = np.arccos(
            np.dot(vector1, vector2) / 
            (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        )
        
        # 라디안을 도로 변환
        angle_deg = np.degrees(angle)
        threshold = self.THRESHOLDS['chin/bottom/angle']
        
        if angle_deg >= threshold:
            value = FeatureValue.ANGULAR
            confidence = min((angle_deg - threshold) / 10 + 0.8, 1.0)
        else:
            value = FeatureValue.SMOOTH
            confidence = min((threshold - angle_deg) / 10 + 0.8, 1.0)
        
        return FeatureAnalysis(
            feature_path='chin/bottom/angle',
            value=value,
            confidence=confidence,
            landmarks=[chin['center'], chin['left'], chin['right']]
        )
    
    # ... (다른 분석 메서드들도 유사하게 구현)

def create_feature_directories(base_dir: str):
    """계층적 특징 디렉토리 생성"""
    def create_recursive(current_path: str, structure: Dict[str, Any]):
        for key, value in structure.items():
            new_path = os.path.join(current_path, key)
            os.makedirs(new_path, exist_ok=True)
            
            if isinstance(value, dict):
                create_recursive(new_path, value)
            elif isinstance(value, list):
                for subdir in value:
                    os.makedirs(os.path.join(new_path, subdir), exist_ok=True)
    
    create_recursive(base_dir, FEATURE_HIERARCHY)

def main():
    parser = argparse.ArgumentParser(description='얼굴형 분석 프로그램')
    parser.add_argument('-i', '--input', 
                        type=str, 
                        required=True,
                        help='입력 이미지 경로')
    parser.add_argument('-o', '--output', 
                        type=str, 
                        required=True,
                        help='결과를 저장할 디렉토리 경로')
    parser.add_argument('-d', '--dataset', 
                        type=str, 
                        default='wflw',
                        choices=['wflw', '300wpublic', '300wprivate', 'merlrav'],
                        help='SPIGA 사전 학습 가중치 데이터셋')
    parser.add_argument('--create-dirs',
                        action='store_true',
                        help='특징별 디렉토리 생성')
    
    args = parser.parse_args()
    
    # 특징별 디렉토리 생성
    if args.create_dirs:
        create_feature_directories(args.output)
        return
    
    # 이미지 읽기
    image = cv2.imread(args.input)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {args.input}")
        return
    
    # 얼굴형 분석기 초기화
    analyzer = FaceShapeAnalyzer(args.dataset)
    
    # 랜드마크 추출
    landmarks = analyzer.extract_landmarks(image)
    if landmarks is None:
        print("랜드마크를 찾을 수 없습니다.")
        return
    
    # 모든 특징 분석
    results = analyzer.analyze_all_features(landmarks)
    
    # 결과 저장
    output_path = Path(args.output)
    image_name = Path(args.input).stem
    
    # 분석된 특징별로 결과 저장
    for feature_path, result in results.items():
        # 결과를 해당 특징 디렉토리에 저장
        feature_dir = os.path.join(args.output, feature_path, result.value.name.lower())
        os.makedirs(feature_dir, exist_ok=True)
        
        # 이미지 복사
        cv2.imwrite(os.path.join(feature_dir, f"{image_name}.jpg"), image)
        
        # 분석 결과 저장
        with open(os.path.join(feature_dir, f"{image_name}_analysis.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Feature: {result.feature_path}\n")
            f.write(f"Value: {result.value.name}\n")
            f.write(f"Confidence: {result.confidence:.2f}\n")
            f.write(f"Used landmarks: {result.landmarks}\n")
    
    # 전체 분석 결과를 하나의 JSON 파일로 저장
    import json
    
    analysis_results = {
        feature_path: {
            'value': result.value.name,
            'confidence': result.confidence,
            'landmarks': result.landmarks
        }
        for feature_path, result in results.items()
    }
    
    with open(output_path / f"{image_name}_full_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print("분석 완료!")

if __name__ == "__main__":
    main() 