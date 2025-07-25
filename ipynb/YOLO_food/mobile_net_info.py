import torch
import torchvision.models as models
import urllib.request
import json

def get_imagenet_classes():
    """ImageNet 클래스 라벨을 다운로드하고 반환"""
    try:
        # ImageNet 클래스 라벨 URL
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with urllib.request.urlopen(url) as f:
            classes = [line.decode('utf-8').strip() for line in f.readlines()]
        return classes
    except Exception as e:
        print(f"클래스 라벨 다운로드 실패: {e}")
        return None

def load_mobilenet_v2():
    """MobileNetV2 모델 로드"""
    print("MobileNetV2 모델 다운로드 중...")
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    print("모델 로드 완료!")
    return model

def analyze_model_structure(model):
    """모델 구조 분석"""
    print("\n" + "="*50)
    print("📱 MobileNetV2 모델 구조 분석")
    print("="*50)
    
    # 전체 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    # 분류기 구조 확인
    print(f"\n분류기 출력 크기: {model.classifier[-1].out_features}개 클래스")
    print("\n분류기 구조:")
    for i, layer in enumerate(model.classifier):
        print(f"  {i}: {layer}")

def display_classes(classes):
    """클래스 목록을 카테고리별로 정리해서 출력"""
    if not classes:
        print("클래스 정보를 가져올 수 없습니다.")
        return
    
    print("\n" + "="*50)
    print("🏷️  ImageNet 클래스 목록 (총 1000개)")
    print("="*50)
    
    # 음식 관련 클래스 찾기
    food_keywords = ['burger', 'pizza', 'hot', 'ice', 'coffee', 'tea', 'cake', 
                     'bread', 'sandwich', 'pretzel', 'bagel', 'muffin', 'waffle',
                     'banana', 'apple', 'orange', 'lemon', 'strawberry']
    
    food_classes = []
    animal_classes = []
    vehicle_classes = []
    other_classes = []
    
    for i, class_name in enumerate(classes):
        class_lower = class_name.lower()
        
        # 음식 분류
        if any(keyword in class_lower for keyword in food_keywords):
            food_classes.append((i, class_name))
        # 동물 분류 (간단한 키워드로)
        elif any(word in class_lower for word in ['dog', 'cat', 'bird', 'fish', 'bear', 'elephant', 'lion']):
            animal_classes.append((i, class_name))
        # 차량 분류
        elif any(word in class_lower for word in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']):
            vehicle_classes.append((i, class_name))
        else:
            other_classes.append((i, class_name))
    
    # 음식 클래스 출력
    print(f"\n🍔 음식 관련 클래스 ({len(food_classes)}개):")
    for idx, (class_idx, class_name) in enumerate(food_classes[:15]):  # 상위 15개만
        print(f"  {class_idx:3d}: {class_name}")
    if len(food_classes) > 15:
        print(f"  ... 외 {len(food_classes) - 15}개 더")
    
    # 동물 클래스 출력
    print(f"\n🐕 동물 관련 클래스 ({len(animal_classes)}개):")
    for idx, (class_idx, class_name) in enumerate(animal_classes[:10]):  # 상위 10개만
        print(f"  {class_idx:3d}: {class_name}")
    if len(animal_classes) > 10:
        print(f"  ... 외 {len(animal_classes) - 10}개 더")
    
    # 차량 클래스 출력
    print(f"\n🚗 차량 관련 클래스 ({len(vehicle_classes)}개):")
    for idx, (class_idx, class_name) in enumerate(vehicle_classes):
        print(f"  {class_idx:3d}: {class_name}")
    
    # 전체 클래스 중 일부 샘플 출력
    print(f"\n📋 전체 클래스 샘플 (처음 20개):")
    for i in range(min(20, len(classes))):
        print(f"  {i:3d}: {classes[i]}")
    
    print(f"\n💡 총 {len(classes)}개 클래스가 학습되어 있습니다.")

def find_specific_classes(classes, keywords):
    """특정 키워드가 포함된 클래스 찾기"""
    found_classes = []
    for i, class_name in enumerate(classes):
        for keyword in keywords:
            if keyword.lower() in class_name.lower():
                found_classes.append((i, class_name))
                break
    return found_classes

def test_model_prediction(model, classes):
    """더미 입력으로 모델 테스트"""
    print("\n" + "="*50)
    print("🧪 모델 예측 테스트")
    print("="*50)
    
    # 더미 입력 생성 (배치크기=1, 채널=3, 높이=224, 너비=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        print("더미 이미지에 대한 Top 5 예측:")
        for i in range(5):
            class_idx = top5_catid[i].item()
            confidence = top5_prob[i].item()
            class_name = classes[class_idx] if classes else f"Class_{class_idx}"
            print(f"  {i+1}. {class_name} ({confidence:.4f})")

def main():
    """메인 함수"""
    print("🔍 MobileNetV2 모델 분석 시작")
    print("="*60)
    
    # 1. 모델 로드
    model = load_mobilenet_v2()
    
    # 2. ImageNet 클래스 라벨 가져오기
    print("\nImageNet 클래스 라벨 다운로드 중...")
    classes = get_imagenet_classes()
    
    # 3. 모델 구조 분석
    analyze_model_structure(model)
    
    # 4. 클래스 목록 출력
    display_classes(classes)
    
    # 5. 특정 클래스 검색 (햄버거 관련)
    if classes:
        print("\n" + "="*50)
        print("🔎 햄버거 관련 클래스 검색")
        print("="*50)
        
        burger_keywords = ['burger', 'hamburger', 'cheeseburger']
        burger_classes = find_specific_classes(classes, burger_keywords)
        
        if burger_classes:
            print("발견된 햄버거 관련 클래스:")
            for class_idx, class_name in burger_classes:
                print(f"  {class_idx:3d}: {class_name}")
        else:
            print("햄버거 관련 클래스를 찾을 수 없습니다.")
    
    # 6. 모델 예측 테스트
    test_model_prediction(model, classes)
    
    print("\n" + "="*60)
    print("✅ 분석 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
