import pandas as pd
import os
import shutil
import sys
import pickle
import torch
from types import ModuleType

# ==============================================================================
# 1. [Mocking] (기존과 동일 - RecBole 실행을 위한 준비)
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
# 2. [Pandas] 데이터 직접 깎기 (RecBole이 말을 안 들으니 직접 합니다)
# ==============================================================================
def filter_data_manually():
    original_path = 'data/amazon-data/amazon-data.inter'
    temp_dir = 'data/amazon-temp'
    temp_path = os.path.join(temp_dir, 'amazon-temp.inter')
    
    print(f"📂 Pandas로 원본 데이터 로드 중: {original_path}")
    
    # 1. 로드
    try:
        df = pd.read_csv(original_path, sep='\t')
        # 컬럼명 정리 (item_id:token -> item_id)
        df.columns = [c.split(':')[0] for c in df.columns]
    except:
        print("❌ 원본 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        sys.exit(1)
        
    print(f"   원본 데이터 행 수: {len(df):,}")
    print(f"   원본 아이템 개수 : {df['item_id'].nunique():,}")

    # 2. 필터링 (User >= 5, Item >= 5)
    # RecBole의 기본 로직은 보통 User 필터링 -> Item 필터링 순서입니다.
    
    # (1) User 필터링
    user_cnt = df['user_id'].value_counts()
    valid_users = user_cnt[user_cnt >= 5].index
    df = df[df['user_id'].isin(valid_users)]
    print(f"   📉 유저 필터링(>=5) 후 행 수: {len(df):,}")
    
    # (2) Item 필터링
    item_cnt = df['item_id'].value_counts()
    valid_items = item_cnt[item_cnt >= 5].index
    df = df[df['item_id'].isin(valid_items)]
    
    final_item_count = df['item_id'].nunique()
    print(f"   📉 아이템 필터링(>=5) 후 행 수: {len(df):,}")
    print(f"   🎯 최종 살아남은 아이템 개수: {final_item_count:,}")
    
    # 3. 임시 파일로 저장
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    # RecBole이 읽을 수 있게 헤더 복원 (user_id:token ...)
    df.rename(columns={'user_id': 'user_id:token', 'item_id': 'item_id:token', 'timestamp': 'timestamp:float'}, inplace=True)
    df.to_csv(temp_path, sep='\t', index=False)
    print(f"📦 임시 파일 저장 완료: {temp_path}")
    
    return 'amazon-temp', final_item_count

# ==============================================================================
# 3. RecBole 호출 및 매핑 저장
# ==============================================================================
from recbole.config import Config
from recbole.data import create_dataset

def extract_vocab_final():
    # 1. Pandas로 먼저 깎아낸 데이터셋 준비
    dataset_name, pandas_item_count = filter_data_manually()
    
    print("\n🚀 RecBole 매핑 생성 시작 (이미 필터링된 데이터 사용)...")
    
    # 2. Config 설정
    # 이미 Pandas에서 다 걸러냈으므로, 여기서는 min_inter=0으로 설정해서 그대로 읽게 합니다.
    config_dict = {
        'data_path': 'data/',           
        'dataset': dataset_name,  
        'gpu_id': -1,                   
        'show_progress': False,
        
        # [중요] 이미 필터링 했으므로 RecBole은 건드리지 마라 (0 설정)
        'min_user_inter': 0,
        'min_item_inter': 0,
        
        'train_neg_sample_args': None, 
        'neg_sampling': None,
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']}
    }
    
    # 3. 데이터셋 생성
    config = Config(model='SASRec', config_dict=config_dict)
    dataset = create_dataset(config)
    
    # 4. 매핑 추출
    token2id = dataset.field2token_id['item_id']
    id2token = dataset.field2id_token['item_id']
    
    recbole_count = len(token2id)
    print("-" * 50)
    print(f"✅ 생성 완료!")
    print(f" - Pandas 계산 개수 : {pandas_item_count}")
    print(f" - RecBole 매핑 개수: {recbole_count} (패딩 포함하면 +1 될 수 있음)")
    
    # 5. 검증 (13,225 근처인지)
    target = 13225
    # RecBole은 내부적으로 0번(PAD)을 추가하므로, token2id 길이는 (실제아이템) 또는 (실제+1) 일 수 있음
    # token2id는 보통 [PAD]를 포함하므로 실제 아이템 수 + 1이 됨.
    
    if abs(recbole_count - target) <= 2:
        print(f"🎉 [대성공] 모델 학습 데이터 크기({target})와 일치합니다!")
    else:
        print(f"⚠️ [주의] 아직도 개수가 다릅니다 ({recbole_count} vs {target}).")
        print("   -> Pandas 필터링 순서(User->Item)가 학습 때와 달랐거나, 원본 파일이 다를 수 있습니다.")

    # 6. 저장
    output_path = "data/recbole_vocab.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({'token2id': token2id, 'id2token': id2token}, f)
        
    print(f"✅ 매핑 파일 저장됨: {output_path}")
    
    # 청소
    try:
        shutil.rmtree('data/amazon-temp')
        print("🧹 임시 폴더 삭제 완료")
    except:
        pass

if __name__ == "__main__":
    extract_vocab_final()