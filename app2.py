import streamlit as st
import pandas as pd
import torch
import random
import os
import sys
import pickle
import numpy as np
import re  # 정규표현식 사용 (숫자 추출)

# ------------------------------------------------------------------
# 1. [필수] 모듈 경로 및 라이브러리 가짜 등록 (에러 방지)
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

# RecBole 패치
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
# 2. 모델 및 데이터 로드
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
    # 1. 메타 데이터
    try:
        all_df = pd.read_pickle('data/meta_lookup.pkl')
    except:
        st.error("meta_lookup.pkl 없음")
        return None, None, None

    # 2. 매핑 데이터
    try:
        with open("data/recbole_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        token2id = vocab['token2id']
        id2token = vocab['id2token']
        
        # 배열 -> 딕셔너리 변환 (안전장치)
        if not isinstance(id2token, dict):
            id2token = {i: str(token) for i, token in enumerate(id2token)}
            
    except:
        st.error("recbole_vocab.pkl 없음. get_true_vocab.py를 실행하세요.")
        return None, None, None
        
    return all_df, token2id, id2token

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

    # 안전한 아이템 범위 설정
    safe_n = min(sas_items, tis_items) if sas_items and tis_items else (tis_items or sas_items)
        
    return sas_model, tis_model, tis_maxlen, tis_timespan, safe_n

# ------------------------------------------------------------------
# 3. [추가] 페르소나 데이터 로드 함수
# ------------------------------------------------------------------
def load_persona_history(all_df):
    persona_path = 'data/여자_대학생_새내기.csv' # 파일 경로 (data 폴더에 넣어주세요)
    
    if not os.path.exists(persona_path):
        st.error(f"페르소나 파일이 없습니다: {persona_path}")
        return []

    try:
        df = pd.read_csv(persona_path)
        history = []
        
        for _, row in df.iterrows():
            # 1. 시점 변환: "30일 전" -> 30 (숫자만 추출)
            days_str = str(row['시점'])
            days = int(re.sub(r'[^0-9]', '', days_str))
            
            # 2. 이름으로 아이템 ID 찾기
            item_name = row['상품 선택'] # CSV 컬럼명 확인 필요
            
            # meta_df에서 이름이 일치하는 행 찾기
            matched_row = all_df[all_df['Item_Name'] == item_name]
            
            if not matched_row.empty:
                # 첫 번째 매칭되는 아이템의 ID 사용
                item_id = str(matched_row.iloc[0]['item_id'])
                
                history.append({
                    'item_id': item_id,
                    'name': item_name,
                    'days_ago': days
                })
            else:
                # 매칭 실패 시 로그 (디버깅용)
                print(f"매핑 실패: {item_name}")
                
        return history
        
    except Exception as e:
        st.error(f"페르소나 로드 중 오류: {e}")
        return []

# ------------------------------------------------------------------
# 4. 메인 로직
# ------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Recommendation A/B Test")
    st.title("🛍️ 쇼핑 패턴 기반 추천 A/B 테스트")

    all_df, token2id, id2token = load_data()
    if all_df is None: return
    
    sas_model, tis_model, tis_maxlen, tis_timespan, safe_n = load_models()

    # UI용 데이터 필터링
    valid_tokens = [t for t, i in token2id.items() if i < safe_n]
    ui_df = all_df[all_df['item_id'].astype(str).isin(valid_tokens) & (all_df['purchase_count'] >= 10)].copy()

    if 'history' not in st.session_state: st.session_state['history'] = []

    # --- 사이드바 ---
    st.sidebar.header("🛒 구매 이력 구성")
    
    # [추가됨] 페르소나 로드 버튼
    st.sidebar.subheader("1. 기본 시퀀스 설정")
    if st.sidebar.button("👩‍🎓 여대생 새내기 모드 적용"):
        persona_history = load_persona_history(all_df)
        if persona_history:
            st.session_state['history'] = persona_history
            # 날짜순 정렬 (과거 -> 현재)
            st.session_state['history'].sort(key=lambda x: x['days_ago'], reverse=True)
            st.success("여대생 페르소나 로드 완료!")
            st.rerun()

    st.sidebar.divider()
    
    st.sidebar.subheader("2. 직접 추가하기")
    if ui_df.empty:
        st.error("데이터 없음")
        return

    l1 = st.sidebar.selectbox("대분류", sorted(ui_df['L1'].unique()))
    l2 = st.sidebar.selectbox("중분류", sorted(ui_df[ui_df['L1']==l1]['L2'].unique()))
    items = ui_df[(ui_df['L1']==l1) & (ui_df['L2']==l2)].sort_values(by='purchase_count', ascending=False)
    
    sel_item = st.sidebar.selectbox("상품 선택", options=items.to_dict('records'), 
                                  format_func=lambda x: f"{x['Item_Name']} ({x['purchase_count']}회)")
    days = st.sidebar.number_input("며칠 전?", 0, 365, 0)
    
    if st.sidebar.button("리스트에 추가"):
        st.session_state['history'].append({
            'item_id': str(sel_item['item_id']), 
            'name': sel_item['Item_Name'], 
            'days_ago': days
        })
        st.session_state['history'].sort(key=lambda x: x['days_ago'], reverse=True)

    if st.sidebar.button("초기화 (전체 삭제)"):
        st.session_state['history'] = []
        st.session_state.pop('last_results', None)
        st.rerun()

    # --- Main ---
    st.subheader("📋 현재 시퀀스 (TimeLine)")
    if st.session_state['history']:
        hist_df = pd.DataFrame(st.session_state['history'])
        hist_df['시점'] = hist_df['days_ago'].apply(lambda x: "오늘" if x==0 else f"{x}일 전")
        st.dataframe(hist_df[['시점', 'name']], width=700)
        
        if st.button("추천 결과 보기 (Inference)", type="primary"):
            if len(st.session_state['history']) < 2:
                st.warning("2개 이상 입력하세요.")
            else:
                with st.spinner("AI 분석 중..."):
                    # 1. 입력 변환 (매핑은 하나만 씁니다!)
                    ids, days_list = [], []
                    for h in st.session_state['history']:
                        if h['item_id'] in token2id:
                            internal = token2id[h['item_id']]
                            if internal < safe_n:
                                ids.append(internal)
                                days_list.append(h['days_ago'])
                    
                    if not ids: st.stop()

                    # 2. 텐서 준비
                    seq = ids[-tis_maxlen:]
                    d_seq = days_list[-tis_maxlen:]
                    pad = tis_maxlen - len(seq)
                    input_ts = torch.LongTensor([[0]*pad + seq]).to(DEVICE)
                    
                    # [중요] 길이는 항상 maxlen으로 고정 (끝방을 보게 함)
                    len_ts = torch.LongTensor([tis_maxlen]).to(DEVICE)
                    
                    # 3. 추론
                    res = []
                    
                    # [A] SASRec
                    if sas_model:
                        scores = sas_model.full_sort_predict(Interaction({'item_id_list': input_ts, 'item_length': len_ts}))
                        topk = torch.topk(scores, 10).indices.cpu().numpy()[0]
                        res.append({'name': 'SASRec', 'ids': topk})
                        
                    # [B] TiSASRec
                    if tis_model:
                        t_seq, t_mat = get_tisasrec_input(d_seq, tis_maxlen, tis_timespan)
                        inter = Interaction({
                            'item_id_list': input_ts, 'item_length': len_ts,
                            'timestamp_list': t_seq.to(DEVICE), 'time_matrix': t_mat.to(DEVICE)
                        })
                        scores = tis_model.full_sort_predict(inter)
                        topk = torch.topk(scores, 10).indices.cpu().numpy()[0]
                        res.append({'name': 'TiSASRec', 'ids': topk})

                    # 4. 결과 변환
                    random.shuffle(res)
                    
                    def ids_to_text(ids):
                        lines = []
                        for i in ids:
                            if i==0: continue
                            if i in id2token:
                                raw = id2token[i]
                                row = all_df[all_df['item_id'].astype(str) == raw]
                                if not row.empty:
                                    d = row.iloc[0]
                                    lines.append(f"**[{d['L1']} > {d['L2']}]**\n{d['Item_Name']}")
                                else:
                                    lines.append(f"Unknown ({raw})")
                            else:
                                lines.append(f"Unknown ID {i}")
                        return lines

                    st.session_state['last_results'] = [
                        {'name': r['name'], 'texts': ids_to_text(r['ids'])} for r in res
                    ]

    if 'last_results' in st.session_state:
        st.divider()
        c1, c2 = st.columns(2)
        r = st.session_state['last_results']
        
        with c1:
            st.info("### 결과 A")
            for i, t in enumerate(r[0]['texts']): st.markdown(f"{i+1}. {t}")
            if st.button("👍 A 승리"): st.success(f"승자: {r[0]['name']}")
            
        with c2:
            st.info("### 결과 B")
            for i, t in enumerate(r[1]['texts']): st.markdown(f"{i+1}. {t}")
            if st.button("👍 B 승리"): st.success(f"승자: {r[1]['name']}")

if __name__ == "__main__":
    main()