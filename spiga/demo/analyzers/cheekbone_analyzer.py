import numpy as np
from typing import List, Dict, Optional
from ..face_shape_analyzer import FeatureAnalysis, FeatureValue

def analyze_cheekbone_prominence(landmarks: np.ndarray, 
                               landmark_groups: Dict, 
                               thresholds: Dict) -> FeatureAnalysis:
    """광대 돌출 정도 분석
    
    Note:
        정확한 광대 돌출 분석을 위해서는 3D 정보나 측면 이미지가 필요합니다.
        현재는 정면 이미지만으로 근사치를 계산합니다.
    """
    cheekbone = landmark_groups['cheekbone']
    chin = landmark_groups['chin']['jawline']
    
    # 광대 너비
    cheekbone_width = np.linalg.norm(
        landmarks[cheekbone['right']] - landmarks[cheekbone['left']]
    )
    
    # 턱선 너비
    jaw_width = np.linalg.norm(
        landmarks[chin['right']] - landmarks[chin['left']]
    )
    
    # 광대/턱선 너비 비율로 돌출 정도 추정
    ratio = cheekbone_width / jaw_width
    
    # 임시 임계값 (실제 데이터로 조정 필요)
    threshold = 1.2  # 광대가 턱선보다 20% 이상 넓으면 돌출로 판단
    
    if ratio >= threshold:
        value = FeatureValue.PROMINENT
        confidence = min((ratio - threshold) / 0.2 + 0.8, 1.0)
    else:
        value = FeatureValue.NORMAL
        confidence = min((threshold - ratio) / 0.2 + 0.8, 1.0)
    
    return FeatureAnalysis(
        feature_path='cheekbone/prominence',
        value=value,
        confidence=confidence,
        landmarks=[cheekbone['left'], cheekbone['right'],
                  chin['left'], chin['right']]
    ) 