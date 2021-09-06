# Mask_Classification
# [개인회고] P-Stage 1 Image Classification

![Untitled (18)](https://user-images.githubusercontent.com/30318926/132158191-e4307715-44d1-4e23-b149-224fbae42886.png)

# 프로젝트의 목표


나만의 개인 모델을 구현해보고 여러가지 기법들을 직접 코드를 짜서 적용해보고 싶었다.

# 이번 프로젝트 진행사항

# **[Model]**

## **사용한 것들**

### **ResNet**

- 18,50,101,152 다양한 model들을 사용을 해봤으나 18에서 가장 좋은 성능을 보여줬습ㄴ디ㅏ.
깊은 layer를 가진 model 사용 시 **Overfitting**이 일어나는 것 같았습니다.
- **ResNet18** 에서 최고기록을 세웠습니다 → 운이 조금 들어갔습니다.

### EfficientNet

- 이 모델을 사용할 때 쯤 model이 크게 중요하지 않은것을 알게 되어 **EfficientNet_b3**를 그냥 사용했습니다.
- 이유로는 속도도 빠르고 성능이 가장 안정적으로 나와 다른 기법들을 테스트하기 좋았음

## 하고싶었으나 못한 것들

### Transformer 기반 model들

- 단순 적용시 성능이 너무 떨어졌습니다.
- 다른 기법이 필요한것인지 확인을 해봐야겠습니다.

→dataset 충분해야만 성능이 나온다.

# [Dataset]

## **사용한 것들**

### FaceNet

- 토론게시판에서 GradCam으로 model이 어느 부분을 보는지 알려주는 글을 본 뒤에 얼굴만 본다는 것을 확인했습니다.
- 외부 배경이나 옷 같은 부분은 노이즈로 작용하는것으로 판단해서 얼굴만 Crop해야 한다 판단했습니다.
- RetinaFace를 사용할려 하였으나 속도 문제로 FaceNet을 사용하였습니다.
- 나중에 김현수님이 RetinaFace로 Crop하신 Dataset으로 훈련 해봤을 때 오히려 성능이 떨어져 FaceNet 으로 Crop한 Dataset을 계속 사용하였습니다.

### **59세 → 60세로 labeling**

- **60세이상 data가 너무 적어** 59세를 60세로 포함시켜봤는데 성능 향상에 도움이 되었습니다.

### M**islabeled**

- 수작업으로 고쳤습니다.

### Augmentation

- Special Mission을 참고하였습니다.
- HorizontalFlip, ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast, GaussNoise 기법을 적용하였습니다.
- 일반화에 도움이 되었습니다.

## 하고싶었으나 못한 것들

### Pseudo-labeling

- eval dataset에서 시도하려 했으나 Cheating 같아서 시도하지 않았습니다.

### CutMix

### Validation 균등하게 나누기

### TTA

→ 이것들은 제 코딩실력 + 시간 문제로 시도해보지 못했습니다.

# [Loss Function]

### Focal Loss

- 대회 중반부터 계속 사용한 Loss Function입니다.
- **Data Imbalance**가 심하기 때문에 일반 CrossEntropy 대신에 사용하였습니다.
- weight는 ins_weight를 사용하였습니다.

![Untitled (19)](https://user-images.githubusercontent.com/30318926/132158235-7857a92a-fcc7-4946-8b61-87e745119cbb.png)

### Cross Entropy Loss

- 초반에 사용하던 Loss Function
- weight를 적용시킨 뒤 눈에띄는 성능 향상이 있었음. → data imbalance가 커보였음.

# [Optimizer]

### Adam

- 가장 무난한 것으로 선택
- Scheduler로는 CosineAnnealing을 사용하였으나 6~7 epochs에서 최고 성능이 나와서 딱히 역할은 못해준것 같다.

# 좋았던 점

- 드디어 notebook 형식에서 마구잡이식으로 하던 방식을 벗어났다! → 쉘에서 명령어만 바꾸면 돌아가게끔 프로젝트 형식으로 변환 성공

![Untitled (20)](https://user-images.githubusercontent.com/30318926/132158244-989c60d4-c041-4cac-a7c3-fe71cbfe03fd.png)

- 개인적으로 이것저것 시도해 볼 수 있어서 좋았다.
- 성능이 올라가는 모습을 보면서 성장하는 느낌을 받았다.
- weight & bias 를 이용하여 logging 하는 법을 직접 적용시킬 수 있어서 좋았다.

# 아쉬웠던 점

- 팀원과의 협업이 조금 부족해서 아쉬웠다.
- 이것저것 적용하고 싶은 것들이 더욱 많았지만 집중력+실력 문제로 시간이 부족하였다.
- 조금 더 깔끔하게 코드를 짜고 싶었다.
- EDA의 중요성을 간과한것 같았다.

# 다음 P-Stage에선..

- 베이스라인 코드 기반으로 협업하는 방식을 적용시켜야겠다.
- 조금 더 깔끔하게 py기반 프로젝트 형식으로 코드를 짜야겠다.
- EDA 과정을 간단하게 생각하는것이 아닌 data를 바탕으로 어떻게 모델을 구현해야할까 까지도 생각해야겠다.
