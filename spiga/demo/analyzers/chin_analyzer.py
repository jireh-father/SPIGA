import numpy as np
from typing import List, Dict, Optional
from ..face_shape_analyzer import FeatureAnalysis, FeatureValue

def analyze_chin_bottom_width(landmarks: np.ndarray, 
                            landmark_groups: Dict, 
                            thresholds: Dict) -> FeatureAnalysis:
    """아래턱 너비 분석"""
    chin = landmark_groups['chin']['bottom']
    
    # 턱 양끝 거리 계산
    width = np.linalg.norm(
        landmarks[chin['right']] - landmarks[chin['left']]
    )
    
    # 기준 너비 (예: 광대 너비)와 비교
    cheekbone_width = np.linalg.norm(
        landmarks[landmark_groups['cheekbone']['right']] -
        landmarks[landmark_groups['cheekbone']['left']]
    )
    
    ratio = width / cheekbone_width
    thresholds = thresholds['chin/width']
    
    # 값과 신뢰도 결정
    if ratio >= thresholds['wide']:
        value = FeatureValue.WIDE
        confidence = min((ratio - thresholds['wide']) / 0.2 + 0.8, 1.0)
    else:
        value = FeatureValue.NARROW
        confidence = min((thresholds['wide'] - ratio) / 0.2 + 0.8, 1.0)
    
    return FeatureAnalysis(
        feature_path='chin/bottom/width',
        value=value,
        confidence=confidence,
        landmarks=[chin['left'], chin['right'], 
                  landmark_groups['cheekbone']['left'],
                  landmark_groups['cheekbone']['right']]
    )

def analyze_chin_bottom_height(landmarks: np.ndarray, 
                             landmark_groups: Dict, 
                             thresholds: Dict) -> FeatureAnalysis:
    """아래턱 높이 분석"""
    chin = landmark_groups['chin']['bottom']
    
    # 입술밑-턱끝 길이 계산 (임시로 이마-턱끝 길이의 비율로 계산)
    chin_height = np.linalg.norm(
        landmarks[chin['center']] - landmarks[landmark_groups['forehead']['top']]
    )
    
    # 전체 얼굴 길이 대비 비율 계산
    face_height = np.linalg.norm(
        landmarks[landmark_groups['forehead']['top']] - 
        landmarks[chin['center']]
    )
    
    ratio = chin_height / face_height
    thresholds = thresholds['chin/bottom/height']
    
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
        landmarks=[chin['center'], landmark_groups['forehead']['top']]
    )

def analyze_chin_bottom_angle(landmarks: np.ndarray, 
                            landmark_groups: Dict, 
                            thresholds: Dict) -> FeatureAnalysis:
    """아래턱 각도 분석"""
    chin = landmark_groups['chin']['bottom']
    
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
    threshold = thresholds['chin/bottom/angle']
    
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

def analyze_chin_width(landmarks: np.ndarray, 
                      landmark_groups: Dict, 
                      thresholds: Dict) -> FeatureAnalysis:
    """턱 전체 너비 분석"""
    chin = landmark_groups['chin']['jawline']
    
    # 턱선 양끝 거리 계산
    width = np.linalg.norm(
        landmarks[chin['right']] - landmarks[chin['left']]
    )
    
    # 기준 너비 (광대 너비)와 비교
    cheekbone_width = np.linalg.norm(
        landmarks[landmark_groups['cheekbone']['right']] -
        landmarks[landmark_groups['cheekbone']['left']]
    )
    
    ratio = width / cheekbone_width
    thresholds = thresholds['chin/width']
    
    if ratio >= thresholds['widest']:
        value = FeatureValue.WIDEST
        confidence = min((ratio - thresholds['widest']) / 0.2 + 0.8, 1.0)
    elif ratio >= thresholds['wide']:
        value = FeatureValue.WIDE
        confidence = min((ratio - thresholds['wide']) / 0.2 + 0.8, 1.0)
    elif ratio >= thresholds['normal']:
        value = FeatureValue.NORMAL
        confidence = 1.0 - abs(ratio - thresholds['normal']) / 0.2
    else:
        value = FeatureValue.NARROW
        confidence = min((thresholds['normal'] - ratio) / 0.2 + 0.8, 1.0)
    
    return FeatureAnalysis(
        feature_path='chin/width',
        value=value,
        confidence=confidence,
        landmarks=[chin['left'], chin['right'],
                  landmark_groups['cheekbone']['left'],
                  landmark_groups['cheekbone']['right']]
    )

def analyze_chin_angle(landmarks: np.ndarray, 
                      landmark_groups: Dict, 
                      thresholds: Dict) -> FeatureAnalysis:
    """턱선 각도 분석"""
    # 구현 필요
    pass

def analyze_chin_height(landmarks: np.ndarray, 
                       landmark_groups: Dict, 
                       thresholds: Dict) -> FeatureAnalysis:
    """턱 전체 높이 분석"""
    # 구현 필요
    pass

def analyze_chin_profile(landmarks: np.ndarray, 
                        landmark_groups: Dict, 
                        thresholds: Dict) -> FeatureAnalysis:
    """턱 측면 프로파일 분석"""
    # 구현 필요
    pass 