import sys
import os
import shutil
import torch
import pickle
from types import ModuleType

# ==============================================================================
# 1. [필수] 라이브러리 및 모델 Mocking (에러 방지)
# ==============================================================================
def mock_lib_with_class(module_name, class_names=[]):
    if module_name in sys.modules: return sys.modules[module_name]
    m = ModuleType(module_name)
    sys.modules[module_name] = m
    for cls_name in class_names:
        if not hasattr(m, cls_name): setattr(m, cls_name, type(cls_name, (object,), {}))
    return m

# 의존성 라이브러리 가짜 등록
for lib in ['kmeans_pytorch', 'lightgbm', 'xgboost', 'ray', 'hyperopt', 'colorama']:
    mock_lib_with_class(lib)
    sys.modules[f"{lib}.sklearn"] = mock_lib_with_class(f"{lib}.sklearn")

if 'kmeans_pytorch' in sys.modules:
    sys.modules['kmeans_pytorch'].kmeans = lambda *args, **kwargs: (None, None)

# RecBole 내부 모델 및 TiSASRec 경로 연결
mock_lib_with_class('recbole.model.general_recommender.ldiffrec', ['LDiffRec'])
mock_lib_with_class('recbole.model.general_recommender.diffrec', ['DiffRec'])

import tisasrec_local
sys.modules['TiSASRec'] = tisasrec_local

# RecBole Utils 패치
import recbole
if not hasattr(recbole, 'utils'):
    recbole.utils = ModuleType('recbole.utils')
    sys.modules['recbole.utils'] = recbole.utils
if not hasattr(recbole.utils, 'enum_type'):
    m_enum = ModuleType('recbole.utils.enum_type')
    sys.modules['recbole.utils.enum_type'] = m_enum
    recbole.utils.enum_type = m_enum
    class Dummy:
        def __init__(self, *args, **kwargs): pass
    for cls in ['ModelType', 'DataLoaderType', 'KGDataLoaderState', 'EvaluatorType', 'InputType', 'FeatureType']:
        setattr(m_enum, cls, Dummy)

# ==============================================================================
# 2. 데이터 경로 확인 및 보정
# ==============================================================================
def prepare_data_folder():
    dataset_name = 'amazon-data' 
    filename = f'{dataset_name}.inter'
    
    current_path = os.path.join('data', filename)
    target_dir = os.path.join('data', dataset_name)
    target_path = os.path.join(target_dir, filename)
    
    if os.path.exists(target_path): return dataset_name
    if os.path.exists(current_path):
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(current_path, target_path)
        return dataset_name
        
    print(f"❌ '{filename}' 파일을 찾을 수 없습니다.")
    sys.exit(1)

# ==============================================================================
# 3. 유저 매핑 추출 로직
# ==============================================================================
from recbole.config import Config
from recbole.data import create_dataset

def extract_user_vocab():
    dataset_name = prepare_data_folder()
    
    print("🚀 유저 매핑(User ID Mapping) 복원 시작...")
    
    # 1. 모델 설정 로드
    pth_path = 'data/TiSASRec-Nov-28-2025_09-45-58.pth'
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    saved_config = checkpoint['config']
    
    # 2. 필터링 조건 설정 (학습 때와 동일하게 5회)
    # 아이템 개수를 맞췄던 그 조건과 동일해야 유저 ID도 맞습니다.
    min_user = 5
    min_item = 5
    max_len = saved_config['MAX_ITEM_LIST_LENGTH'] if 'MAX_ITEM_LIST_LENGTH' in saved_config else 50

    config_dict = {
        'data_path': 'data/',           
        'dataset': dataset_name,  
        'gpu_id': -1,                   
        'show_progress': False,
        'min_user_inter': min_user,
        'min_item_inter': min_item,
        'MAX_ITEM_LIST_LENGTH': max_len,
        'train_neg_sample_args': None, 
        'neg_sampling': None,
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']}
    }
    
    print(f"ℹ️ 필터링 조건: User >= {min_user}, Item >= {min_item}")
    
    # 3. 데이터셋 생성
    config = Config(model='SASRec', config_dict=config_dict)
    dataset = create_dataset(config)
    
    # 4. 유저 매핑 추출 (여기가 핵심!)
    # dataset.field2token_id['user_id'] : 원본 유저ID -> 모델 유저ID
    # dataset.field2id_token['user_id'] : 모델 유저ID -> 원본 유저ID
    user_token2id = dataset.field2token_id['user_id']
    user_id2token = dataset.field2id_token['user_id']
    
    print("-" * 50)
    print(f"✅ 추출 완료!")
    print(f" - 총 유저 수 (모델 학습 기준): {len(user_token2id)}")
    
    # 5. 저장
    output_path = "data/user_vocab.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({
            'user_token2id': user_token2id,
            'user_id2token': user_id2token
        }, f)
        
    print(f"✅ 유저 매핑 파일 저장됨: {output_path}")

if __name__ == "__main__":
    extract_user_vocab()