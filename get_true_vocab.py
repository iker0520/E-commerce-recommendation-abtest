import sys
import os
import shutil
import torch
import pickle
from types import ModuleType

# ==============================================================================
# 1. [Mocking] (기존과 동일)
# ==============================================================================
def mock_lib_with_class(module_name, class_names=[]):
    if module_name in sys.modules: return sys.modules[module_name]
    m = ModuleType(module_name)
    sys.modules[module_name] = m
    for cls_name in class_names:
        if not hasattr(m, cls_name): setattr(m, cls_name, type(cls_name, (object,), {}))
    return m

libs = ['kmeans_pytorch', 'lightgbm', 'xgboost', 'ray', 'hyperopt', 'colorama']
for lib in libs:
    mock_lib_with_class(lib)
    sys.modules[f"{lib}.sklearn"] = mock_lib_with_class(f"{lib}.sklearn")

if 'kmeans_pytorch' in sys.modules:
    sys.modules['kmeans_pytorch'].kmeans = lambda *args, **kwargs: (None, None)

mock_lib_with_class('recbole.model.general_recommender.ldiffrec', ['LDiffRec'])
mock_lib_with_class('recbole.model.general_recommender.diffrec', ['DiffRec'])

import tisasrec_local
sys.modules['TiSASRec'] = tisasrec_local

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
# 2. [핵심] 데이터셋 복사 및 이름 변경 (캐시 우회 전략)
# ==============================================================================
def prepare_new_dataset():
    # 원본 (기존에 쓰던 것)
    old_name = 'amazon-data'
    old_path = os.path.join('data', old_name, f'{old_name}.inter')
    
    # 신규 (새로운 이름)
    new_name = 'amazon-filtered'
    new_dir = os.path.join('data', new_name)
    new_path = os.path.join(new_dir, f'{new_name}.inter')
    
    # 1. 원본 파일 찾기 (data/amazon-data.inter 또는 data/amazon-data/amazon-data.inter)
    if not os.path.exists(old_path):
        # 혹시 data 폴더 바로 밑에 있는지 확인
        alt_path = os.path.join('data', f'{old_name}.inter')
        if os.path.exists(alt_path):
            old_path = alt_path
        else:
            print(f"❌ 원본 데이터 파일을 찾을 수 없습니다: {old_path}")
            sys.exit(1)
            
    # 2. 새로운 폴더 만들고 파일 복사
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        
    print(f"📦 캐시 회피를 위해 데이터 복사 중...")
    print(f"   {old_path} -> {new_path}")
    shutil.copy(old_path, new_path)
    
    return new_name

# ==============================================================================
# 3. 매핑 생성 로직
# ==============================================================================
from recbole.config import Config
from recbole.data import create_dataset

def extract_vocab_force():
    # 1. 새로운 이름의 데이터셋 준비
    dataset_name = prepare_new_dataset()
    
    print("🚀 RecBole 데이터셋 생성 시작 (필터링 적용)...")
    
    # 2. 모델 설정 로드
    pth_path = 'data/TiSASRec-Nov-28-2025_09-45-58.pth'
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    saved_config = checkpoint['config']
    
    # 3. 필터링 조건 설정 (5회 이상)
    min_user = 5
    min_item = 5
    max_len = saved_config.get('MAX_ITEM_LIST_LENGTH', 50) if hasattr(saved_config, 'get') else 50

    config_dict = {
        'data_path': 'data/',           
        'dataset': dataset_name,  # 'amazon-filtered' (새 이름!)
        'gpu_id': -1,                   
        'show_progress': False,
        
        # [중요] 필터링 조건
        'min_user_inter': min_user,
        'min_item_inter': min_item,
        'MAX_ITEM_LIST_LENGTH': max_len,
        
        'train_neg_sample_args': None, 
        'neg_sampling': None,
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']}
    }
    
    print(f"ℹ️ 필터링 조건: User >= {min_user}, Item >= {min_item}")
    
    # 4. 데이터셋 생성
    # 이름이 바뀌었으므로 RecBole은 무조건 처음부터 다시 계산합니다.
    config = Config(model='SASRec', config_dict=config_dict)
    dataset = create_dataset(config)
    
    token2id = dataset.field2token_id['item_id']
    id2token = dataset.field2id_token['item_id']
    
    count = len(token2id)
    print("-" * 50)
    print(f"✅ 생성 완료!")
    print(f" - 매핑된 아이템 개수: {count}")
    
    target = 13225
    if abs(count - target) <= 1:
        print(f"🎉 [성공] 모델 학습 데이터 크기({target})와 일치합니다!")
    else:
        print(f"⚠️ [주의] 개수가 다릅니다 ({count} vs {target}).")
        print("   -> 원본 데이터 파일 자체가 학습 때와 다른 파일일 수 있습니다.")

    # 5. 저장
    output_path = "data/recbole_vocab.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({'token2id': token2id, 'id2token': id2token}, f)
        
    print(f"✅ 매핑 파일 저장됨: {output_path}")
    
    # 청소 (복사한 파일 삭제)
    try:
        shutil.rmtree(os.path.join('data', dataset_name))
        print("🧹 임시 데이터 폴더 정리 완료")
    except:
        pass

if __name__ == "__main__":
    extract_vocab_force()