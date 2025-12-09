import streamlit as st
import pandas as pd
import torch
import random
import os
import sys
import pickle
import numpy as np
import re
import glob

# ------------------------------------------------------------------
# 1. [필수] 모듈 경로 및 라이브러리 가짜 등록 (기존 유지)
# ------------------------------------------------------------------
import tisasrec_local
sys.modules['TiSASRec'] = tisasrec_local

from types import ModuleType
def mock_lib(name):
    if name not in sys.modules: sys.modules[name] = ModuleType(name)
    return sys.modules[name]

for lib in ['kmeans_pytorch', 'lightgbm', 'xgboost', 'ray', 'hyperopt', 'colorama']:
    mock_lib(lib)
    sys.modules[f"{lib}.sklearn"] = mock_lib(f"{lib}.sklearn")

if 'kmeans_pytorch' in sys.modules:
    sys.modules['kmeans_pytorch'].kmeans = lambda *args, **kwargs: (None, None)

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

# ------------------------------------------------------------------
# 2. 데이터 및 모델 로드
# ------------------------------------------------------------------
from recbole.model.sequential_recommender.sasrec import SASRec
from tisasrec_local import TiSASRec
from recbole.data.interaction import Interaction
from utils import get_tisasrec_input

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MockDataset:
    def __init__(self, n_items):
        self.n_items = n_items
    def num(self, field):
        return self.n_items

@st.cache_data
def load_data():
    try:
        all_df = pd.read_pickle('data/meta_lookup.pkl')
    except:
        st.error("data/meta_lookup.pkl 파일이 없습니다.")
        return None, None, None

    try:
        with open("data/recbole_vocab.pkl", "rb") as f:
            vocab_tis = pickle.load(f)
        if not isinstance(vocab_tis['id2token'], dict):
            vocab_tis['id2token'] = {i: str(t) for i, t in enumerate(vocab_tis['id2token'])}
    except:
        st.error("recbole_vocab.pkl (TiSASRec용) 없음")
        return None, None, None, None, None

    try:
        with open("data/sasrec_vocab.pkl", "rb") as f:
            vocab_sas = pickle.load(f)
        if not isinstance(vocab_sas['id2token'], dict):
            vocab_sas['id2token'] = {i: str(t) for i, t in enumerate(vocab_sas['id2token'])}
    except:
        st.error("sasrec_vocab.pkl (SASRec용) 없음")
        return None, None, None, None, None
        
    return all_df, vocab_tis, vocab_sas

@st.cache_resource
def load_models():
    # SASRec
    sas_path = 'data/SASRec-Nov-27-2025_10-12-11.pth'
    sas_model, sas_items = None, 0
    try:
        ckpt = torch.load(sas_path, map_location=DEVICE, weights_only=False)
        sas_items = ckpt['state_dict']['item_embedding.weight'].shape[0]
        sas_model = SASRec(ckpt['config'], MockDataset(sas_items)).to(DEVICE)
        sas_model.load_state_dict(ckpt['state_dict'])
        sas_model.eval()
    except Exception as e:
        st.warning(f"SASRec 로드 실패: {e}")

    # TiSASRec
    tis_path = 'data/TiSASRec-Nov-28-2025_09-45-58.pth'
    tis_model, tis_items = None, 0
    tis_maxlen, tis_timespan = 50, 256
    try:
        ckpt = torch.load(tis_path, map_location=DEVICE, weights_only=False)
        tis_items = ckpt['state_dict']['item_embedding.weight'].shape[0]
        conf = ckpt['config']
        tis_model = TiSASRec(conf, MockDataset(tis_items)).to(DEVICE)
        tis_model.load_state_dict(ckpt['state_dict'])
        tis_model.eval()
        tis_maxlen = conf['MAX_ITEM_LIST_LENGTH']
        tis_timespan = conf['time_span']
    except Exception as e:
        st.error(f"TiSASRec 로드 실패: {e}")

    safe_n = min(sas_items, tis_items) if sas_items and tis_items else (tis_items or sas_items)
        
    return sas_model, tis_model, tis_maxlen, tis_timespan, safe_n

# ------------------------------------------------------------------
# 3. [기능 업그레이드] 페르소나 데이터 로드 함수 (폴더 경로 수정 반영)
# ------------------------------------------------------------------
def load_persona_history(all_df, filename):
    # data/personas 폴더 안에서 파일을 찾습니다.
    persona_path = os.path.join('data', 'personas', filename)
    
    if not os.path.exists(persona_path):
        st.error(f"❌ 파일을 찾을 수 없습니다: {persona_path}")
        return []

    try:
        # 한글 인코딩 호환성 처리
        try:
            df = pd.read_csv(persona_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(persona_path, encoding='cp949')

        history = []
        for _, row in df.iterrows():
            # 1. 시점 변환: "30일 전" -> 30 (숫자만 추출)
            days_str = str(row.get('시점', '0'))
            days_match = re.search(r'\d+', days_str)
            days = int(days_match.group()) if days_match else 0
            
            # 2. 아이템 이름 확인 (csv 컬럼명 대응)
            item_name = row.get('상품 선택') or row.get('name')
            if not item_name: continue

            # 3. meta_df에서 ID 찾기
            matched_row = all_df[all_df['Item_Name'] == item_name]
            
            if not matched_row.empty:
                item_id = str(matched_row.iloc[0]['item_id'])
                history.append({
                    'item_id': item_id,
                    'name': item_name,
                    'days_ago': days
                })
            else:
                # 매핑 실패 시 로깅 (UI에는 띄우지 않음)
                print(f"매핑 실패: {item_name}")
                
        return history
    except Exception as e:
        st.error(f"페르소나 로드 오류 ({filename}): {e}")
        return []

# ------------------------------------------------------------------
# 4. 메인 로직
# ------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Recommendation Model A/B Test")
    st.title("🛍️ 쇼핑 패턴 기반 추천 A/B Test")

    all_df, vocab_tis, vocab_sas = load_data()
    if all_df is None: return
    
    sas_model, tis_model, tis_maxlen, tis_timespan, safe_n = load_models()

    # UI 필터링
    valid_tokens = [t for t, i in vocab_tis['token2id'].items() if i < safe_n]
    ui_df = all_df[all_df['item_id'].astype(str).isin(valid_tokens) & (all_df['purchase_count'] >= 10)].copy()

    # 세션 상태 초기화
    if 'history' not in st.session_state: st.session_state['history'] = []

    # ---------------- Sidebar: 입력 UI ----------------
    st.sidebar.header("🛒 구매 이력 구성")
    
    # [1] 페르소나 선택 섹션 (업데이트: 폴더 스캔 및 선택 안함 옵션)
    st.sidebar.subheader("1. 페르소나 선택")
    
    # personas 폴더 자동 생성 및 파일 스캔
    persona_dir = os.path.join('data', 'personas')
    if not os.path.exists(persona_dir):
        os.makedirs(persona_dir, exist_ok=True)
        
    persona_files = [f for f in os.listdir(persona_dir) if f.endswith('.csv')]
    
    # '선택 안 함' 옵션을 맨 앞에 추가
    options = ["직접 입력 (선택 안 함)"] + persona_files
    
    selected_persona = st.sidebar.selectbox("테스터 유형을 선택하세요:", options)
    
    # 파일을 선택했을 때만 로드 버튼 표시
    if selected_persona != "직접 입력 (선택 안 함)":
        if st.sidebar.button("📂 선택한 페르소나 불러오기"):
            persona_history = load_persona_history(all_df, selected_persona)
            if persona_history:
                st.session_state['history'] = persona_history
                # 날짜 내림차순 정렬 (최신이 위로)
                st.session_state['history'].sort(key=lambda x: x['days_ago'], reverse=True)
                st.success(f"'{selected_persona}' 로드 완료! ({len(persona_history)}개 아이템)")
                st.session_state.pop('last_results', None) # 기존 결과 초기화
                st.rerun()

    st.sidebar.divider()
    
    # [2] 직접 추가 섹션
    st.sidebar.subheader("2. 아이템 직접 추가")
    if ui_df.empty:
        st.error("표시할 아이템 데이터가 없습니다.")
        return

    l1 = st.sidebar.selectbox("대분류", sorted(ui_df['L1'].unique()))
    l2 = st.sidebar.selectbox("중분류", sorted(ui_df[ui_df['L1']==l1]['L2'].unique()))
    items = ui_df[(ui_df['L1']==l1) & (ui_df['L2']==l2)].sort_values(by='purchase_count', ascending=False)
    
    sel_item = st.sidebar.selectbox("상품 선택", options=items.to_dict('records'), 
                                  format_func=lambda x: f"{x['Item_Name']} ({x['purchase_count']}회)")
    days = st.sidebar.number_input("며칠 전 구매했나요?", 0, 365, 0)
    
    if st.sidebar.button("➕ 리스트에 추가"):
        st.session_state['history'].append({
            'item_id': str(sel_item['item_id']), 
            'name': sel_item['Item_Name'], 
            'days_ago': days
        })
        st.session_state['history'].sort(key=lambda x: x['days_ago'], reverse=True)
        st.session_state.pop('last_results', None) # 결과 초기화
        st.rerun()

    if st.sidebar.button("🗑️ 전체 초기화"):
        st.session_state['history'] = []
        st.session_state.pop('last_results', None)
        st.rerun()

    # ---------------- Main: 시퀀스 관리 및 추론 ----------------
    st.subheader("📋 현재 시퀀스 (TimeLine)")
    
    if not st.session_state['history']:
        st.info("👈 좌측 사이드바에서 페르소나를 선택하거나, 아이템을 직접 추가해주세요.")
    else:
        # [기능 추가] 시퀀스 목록 및 개별 삭제 기능 구현
        st.markdown("---")
        # enumerate를 사용하여 인덱스를 확보 (삭제 시 필요)
        for i, item in enumerate(st.session_state['history']):
            col1, col2, col3 = st.columns([1, 6, 1])
            
            # 시간 표시 텍스트
            if item['days_ago'] == 0:
                time_str = "오늘"
            else:
                time_str = f"{item['days_ago']}일 전"
            
            with col1:
                st.caption(time_str)
            with col2:
                st.write(f"**{item['name']}**")
            with col3:
                # 삭제 버튼: 고유 key를 부여하여 충돌 방지
                if st.button("❌", key=f"del_{i}", help="이 아이템만 삭제"):
                    st.session_state['history'].pop(i)
                    st.session_state.pop('last_results', None) # 결과 초기화
                    st.rerun()
        st.markdown("---")
        
        # 추론 버튼
        if st.button("🚀 추천 결과 비교 (Model A vs B)", type="primary"):
            if len(st.session_state['history']) < 2:
                st.warning("정확한 분석을 위해 아이템을 2개 이상 입력해주세요.")
            else:
                with st.spinner("두 모델이 시퀀스를 분석 중입니다..."):
                    # --- 입력 데이터 준비 ---
                    t2i_tis = vocab_tis['token2id']
                    ids_tis, days_list = [], []
                    for h in st.session_state['history']:
                        if h['item_id'] in t2i_tis:
                            ids_tis.append(t2i_tis[h['item_id']])
                            days_list.append(h['days_ago'])
                    
                    t2i_sas = vocab_sas['token2id']
                    ids_sas = []
                    for h in st.session_state['history']:
                        if h['item_id'] in t2i_sas:
                            ids_sas.append(t2i_sas[h['item_id']])

                    if not ids_tis or not ids_sas:
                        st.error("매핑 가능한 아이템이 하나도 없습니다.")
                        st.stop()

                    # --- 추론 실행 ---
                    
                    # [Model A] SASRec
                    # SASRec은 시간 정보 없이 아이템 시퀀스만 사용
                    seq_sas = ids_sas[-tis_maxlen:]
                    pad_len_sas = tis_maxlen - len(seq_sas)
                    input_sas = torch.LongTensor([[0]*pad_len_sas + seq_sas]).to(DEVICE)
                    len_sas = torch.LongTensor([tis_maxlen]).to(DEVICE) 
                    
                    topk_A_ids = []
                    if sas_model:
                        inter_sas = Interaction({'item_id_list': input_sas, 'item_length': len_sas})
                        scores_A = sas_model.full_sort_predict(inter_sas)
                        scores_A = scores_A.cpu().detach().numpy()[0]
                        topk_A_indices = np.argsort(scores_A)[::-1][:10]
                        topk_A_ids = topk_A_indices.tolist()

                    # [Model B] TiSASRec
                    # TiSASRec은 아이템 시퀀스 + 시간 간격(Interval) 정보 사용
                    seq_tis = ids_tis[-tis_maxlen:]
                    d_seq = days_list[-tis_maxlen:]
                    pad_len_tis = tis_maxlen - len(seq_tis)
                    input_tis = torch.LongTensor([[0]*pad_len_tis + seq_tis]).to(DEVICE)
                    len_tis = torch.LongTensor([tis_maxlen]).to(DEVICE)
                    
                    # 시간 매트릭스 계산 (utils.py 의존)
                    t_seq, t_mat = get_tisasrec_input(d_seq, tis_maxlen, tis_timespan)
                    
                    topk_B_ids = []
                    if tis_model:
                        inter_tis = Interaction({
                            'item_id_list': input_tis, 'item_length': len_tis,
                            'timestamp_list': t_seq.to(DEVICE), 'time_matrix': t_mat.to(DEVICE)
                        })
                        scores_B = tis_model.full_sort_predict(inter_tis)
                        scores_B = scores_B.cpu().detach().numpy()[0]
                        topk_B_indices = np.argsort(scores_B)[::-1][:10]
                        topk_B_ids = topk_B_indices.tolist()

                    # --- 결과 저장 (순서 랜덤 섞기: Blind Test) ---
                    results_list = [
                        {'ids': topk_A_ids, 'name': 'SASRec', 'type': 'A'},
                        {'ids': topk_B_ids, 'name': 'TiSASRec', 'type': 'B'}
                    ]
                    random.shuffle(results_list)
                    
                    st.session_state['last_results'] = results_list

    # ---------------- 결과 출력 ----------------
    if 'last_results' in st.session_state:
        st.divider()
        st.subheader("🔎 추천 결과 비교 (Blind Test)")
        
        results = st.session_state['last_results']
        res_left = results[0]
        res_right = results[1]
        
        # ID -> 정보 변환 헬퍼 함수
        def get_item_info_detail(internal_id, model_type):
            if model_type == 'A':
                i2t = vocab_sas['id2token']
            else:
                i2t = vocab_tis['id2token']
                
            if internal_id in i2t:
                raw_id = i2t[internal_id]
                row = all_df[all_df['item_id'].astype(str) == raw_id]
                if not row.empty:
                    d = row.iloc[0]
                    return f"{d['L1']} > {d['L2']}", d['Item_Name']
                return "Unknown Cat", "Unknown Name"
            return "-", "-"

        c1, c2 = st.columns(2)
        
        # 왼쪽 결과
        with c1:
            st.info("### Option 1")
            for i, idx in enumerate(res_left['ids']):
                if idx == 0: continue
                cat, name = get_item_info_detail(idx, res_left['type'])
                st.markdown(f"**{i+1}. [{cat}]**\n{name}")
            
            if st.button("👈 Option 1 선택"):
                # 로컬 CSV 저장
                save_log(res_left['name'], res_right['name'], "Option 1")
                st.balloons()
                st.success(f"선택한 모델은 [{res_left['name']}] 입니다!")

        # 오른쪽 결과
        with c2:
            st.success("### Option 2")
            for i, idx in enumerate(res_right['ids']):
                if idx == 0: continue
                cat, name = get_item_info_detail(idx, res_right['type'])
                st.markdown(f"**{i+1}. [{cat}]**\n{name}")
                
            if st.button("👉 Option 2 선택"):
                # 로컬 CSV 저장
                save_log(res_right['name'], res_left['name'], "Option 2")
                st.balloons()
                st.success(f"선택한 모델은 [{res_right['name']}] 입니다!")

# CSV 저장 함수
def save_log(winner_model, loser_model, choice_label):
    log_data = {
        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "winner": winner_model,
        "loser": loser_model,
        "user_choice": choice_label
    }
    file_path = 'ab_test_results.csv'
    df = pd.DataFrame([log_data])
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(file_path, index=False, header=False, mode='a', encoding='utf-8-sig')

if __name__ == "__main__":
    main()