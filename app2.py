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
from recbole.data.interaction import Interaction

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MockDataset:
    def __init__(self, n_items):
        self.n_items = n_items
    def num(self, field):
        return self.n_items

def load_translations():
    """번역 파일 로드 및 매핑 딕셔너리 생성"""
    csv_path = './data/translation_progress.csv'
    if not os.path.exists(csv_path):
        return {}, {}, {}
    
    try:
        df = pd.read_csv(csv_path)
        df['Original_English'] = df['Original_English'].astype(str).str.strip()
        df['Translated_Korean'] = df['Translated_Korean'].astype(str).str.strip()
        
        l1_df = df[df['Category_Type'] == '대분류']
        l1_map = dict(zip(l1_df['Original_English'], l1_df['Translated_Korean']))
        
        l2_df = df[df['Category_Type'] == '중분류']
        l2_map = dict(zip(l2_df['Original_English'], l2_df['Translated_Korean']))
        
        item_df = df[df['Category_Type'] == '선택']
        item_map = dict(zip(item_df['Original_English'], item_df['Translated_Korean']))
        
        return l1_map, l2_map, item_map
    except Exception as e:
        st.error(f"번역 파일 로드 중 오류 발생: {e}")
        return {}, {}, {}

@st.cache_data
def load_data():
    # 1. 메타 데이터
    try:
        all_df = pd.read_pickle('data/meta_lookup.pkl')
    except:
        st.error("data/meta_lookup.pkl 파일이 없습니다.")
        return None, None, None

    # 2. 번역 데이터 적용
    l1_map, l2_map, item_map = load_translations()
    
    all_df['L1'] = all_df['L1'].astype(str).str.strip()
    all_df['L2'] = all_df['L2'].astype(str).str.strip()
    all_df['Item_Name'] = all_df['Item_Name'].astype(str).str.strip()

    all_df['L1_KR'] = all_df['L1'].map(l1_map).fillna(all_df['L1'])
    all_df['L2_KR'] = all_df['L2'].map(l2_map).fillna(all_df['L2'])
    all_df['Item_Name_KR'] = all_df['Item_Name'].map(item_map).fillna(all_df['Item_Name'])

    # 3. 매핑 데이터
    try:
        with open("data/recbole_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        token2id = vocab['token2id']
        id2token = vocab['id2token']
        
        if not isinstance(id2token, dict):
            id2token = {i: str(token) for i, token in enumerate(id2token)}
    except Exception as e:
        st.error(f"매핑 파일 로드 오류: {e}")
        return None, None, None
        
    return all_df, token2id, id2token

@st.cache_data
def load_cycle_data():
    try:
        with open("data/item_cycle_lookup.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return {} 

@st.cache_resource
def load_models():
    sas_path = 'data/SASRec-Nov-27-2025_10-12-11.pth'
    sas_model, sas_n_items = None, 0
    try:
        checkpoint = torch.load(sas_path, map_location=DEVICE, weights_only=False)
        sas_n_items = checkpoint['state_dict']['item_embedding.weight'].shape[0]
        sas_model = SASRec(checkpoint['config'], MockDataset(sas_n_items)).to(DEVICE)
        sas_model.load_state_dict(checkpoint['state_dict'])
        sas_model.eval()
        maxlen = checkpoint['config']['MAX_ITEM_LIST_LENGTH']
    except Exception as e:
        st.warning(f"SASRec 로드 실패: {e}")
        maxlen = 50

    return sas_model, sas_n_items, maxlen

# ------------------------------------------------------------------
# 3. 페르소나 데이터 로드 함수
# ------------------------------------------------------------------
def load_persona_history(all_df, filename):
    persona_path = os.path.join('data', 'personas', filename)
    
    if not os.path.exists(persona_path):
        st.error(f"❌ 파일을 찾을 수 없습니다: {persona_path}")
        return []

    try:
        try:
            df = pd.read_csv(persona_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(persona_path, encoding='cp949')

        history = []
        for _, row in df.iterrows():
            days_str = str(row.get('시점', '0'))
            days_match = re.search(r'\d+', days_str)
            days = int(days_match.group()) if days_match else 0
            
            item_name_raw = row.get('상품 선택') or row.get('name')
            if not item_name_raw: continue

            matched_row = all_df[all_df['Item_Name'] == item_name_raw]
            
            if not matched_row.empty:
                item_id = str(matched_row.iloc[0]['item_id'])
                item_name_kr = matched_row.iloc[0]['Item_Name_KR']
                history.append({
                    'item_id': item_id,
                    'name': item_name_kr,
                    'days_ago': days
                })
            else:
                pass 
                
        return history
    except Exception as e:
        st.error(f"페르소나 로드 중 오류: {e}")
        return []

# ------------------------------------------------------------------
# 4. 로직 함수 (Cycle Filtering)
# ------------------------------------------------------------------
def check_cycle_filtering(days_ago, cycle_info):
    if not cycle_info: return days_ago < 7
    p10 = cycle_info.get('p10', 0)
    p25 = cycle_info.get('p25', 0)
    
    if days_ago < p10: return random.random() < 1
    elif p10 <= days_ago < p25: return random.random() < 0.5
    else: return False

# ------------------------------------------------------------------
# 5. 메인 로직
# ------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Recommendation Rule A/B Test")
    st.title("🛍️ 쇼핑 패턴 기반 추천 알고리즘 A/B Test")

    all_df, token2id, id2token = load_data()
    if all_df is None: return
    
    cycle_data = load_cycle_data()
    sas_model, safe_n_items, maxlen = load_models()

    valid_tokens = [t for t, i in token2id.items() if i < safe_n_items]
    ui_df = all_df[all_df['item_id'].astype(str).isin(valid_tokens) & (all_df['purchase_count'] >= 10)].copy()

    if 'history' not in st.session_state: st.session_state['history'] = []

    # ---------------- Sidebar ----------------
    st.sidebar.header("🛒 구매 이력 구성")
    
    # [1] 페르소나 선택
    st.sidebar.subheader("1. 페르소나 선택")
    persona_dir = os.path.join('data', 'personas')
    if not os.path.exists(persona_dir): os.makedirs(persona_dir, exist_ok=True)
        
    persona_files = [f for f in os.listdir(persona_dir) if f.endswith('.csv')]
    options = ["직접 입력 (선택 안 함)"] + persona_files
    
    # .csv 제거
    selected_persona = st.sidebar.selectbox(
        "테스터 유형을 선택하세요:", 
        options,
        format_func=lambda x: x.replace(".csv", "") if x != "직접 입력 (선택 안 함)" else x
    )
    
    if selected_persona != "직접 입력 (선택 안 함)":
        if st.sidebar.button("📂 선택한 페르소나 불러오기"):
            persona_history = load_persona_history(all_df, selected_persona)
            if persona_history:
                st.session_state['history'] = persona_history
                st.session_state['history'].sort(key=lambda x: x['days_ago'], reverse=True)
                st.success(f"'{selected_persona.replace('.csv','')}' 로드 완료!")
                st.session_state.pop('raw_scores', None)
                st.session_state.pop('ab_mapping', None)
                st.rerun()

    st.sidebar.divider()

    # [2] 직접 추가
    st.sidebar.subheader("2. 아이템 추가")
    if not ui_df.empty:
        l1_list = sorted(ui_df['L1_KR'].unique())
        l1 = st.sidebar.selectbox("대분류", l1_list)
        
        l1_mask = ui_df['L1_KR'] == l1
        l2_list = sorted(ui_df[l1_mask]['L2_KR'].unique())
        l2 = st.sidebar.selectbox("중분류", l2_list)
        
        items = ui_df[l1_mask & (ui_df['L2_KR'] == l2)].sort_values(by='purchase_count', ascending=False)
        
        # 구매횟수 제거, 상품명만 표시
        sel_item = st.sidebar.selectbox(
            "상품 선택", 
            options=items.to_dict('records'), 
            format_func=lambda x: x['Item_Name_KR']
        )

        days = st.sidebar.number_input("며칠 전 구매?", 0, 365, 0)
        
        if st.sidebar.button("➕ 리스트에 추가"):
            st.session_state['history'].append({
                'item_id': str(sel_item['item_id']),
                'name': sel_item['Item_Name_KR'],
                'days_ago': days
            })
            st.session_state['history'].sort(key=lambda x: x['days_ago'], reverse=True)
            st.rerun()

    if st.sidebar.button("🗑️ 전체 초기화"):
        st.session_state['history'] = []
        st.session_state.pop('raw_scores', None) 
        st.session_state.pop('ab_mapping', None) 
        st.rerun()

    # --- Main: 시퀀스 확인 ---
    st.subheader("📋 이커머스 상품 구매 내역")
    st.info("""
        테스터님이 직접 구매 히스토리를 구성하면, 구매주기를 고려한 추천과 그렇지 않은 추천 결과가 제공됩니다.

        최대한 본인의 실제 구매패턴을 기반으로 시퀀스를 자유롭게 작성해주세요!

        왼쪽 사이드바에서 특정 페르소나를 불러오거나, 직접 아이템을 추가할 수 있습니다.
        """)
    
    if not st.session_state['history']:
        st.info("""
        시퀀스를 입력해주세요.
        """)
    else:
        st.markdown("---")
        for i, item in enumerate(st.session_state['history']):
            col1, col2, col3 = st.columns([1, 6, 1])
            time_str = "오늘" if item['days_ago'] == 0 else f"{item['days_ago']}일 전"
            
            with col1: st.caption(time_str)
            with col2: st.write(f"**{item['name']}**")
            with col3:
                # 개별 삭제 기능
                if st.button("❌", key=f"del_{i}"):
                    st.session_state['history'].pop(i)
                    st.session_state.pop('raw_scores', None) 
                    st.session_state.pop('ab_mapping', None)
                    st.rerun()
        st.markdown("---")
    
        # alpha 고정
        alpha = 2.0 
        
        # ------------------------------------------------------------------
        # 추론 버튼
        # ------------------------------------------------------------------
        if st.button("🚀 추천 결과 생성", type="primary"):
            if len(st.session_state['history']) < 2:
                st.warning("아이템을 2개 이상 넣어주세요.")
            else:
                with st.spinner("AI 분석 중..."):
                    if 'ab_mapping' in st.session_state:
                        del st.session_state['ab_mapping']
                    
                    hist_ids = []
                    for h in st.session_state['history']:
                        if h['item_id'] in token2id:
                            internal_id = token2id[h['item_id']]
                            if internal_id < safe_n_items:
                                hist_ids.append(internal_id)
                    
                    if not hist_ids: st.stop()
                        
                    seq_ids = hist_ids[-maxlen:]
                    pad_len = maxlen - len(seq_ids)
                    input_ids = [0] * pad_len + seq_ids
                    
                    item_seq = torch.LongTensor([input_ids]).to(DEVICE)
                    item_len = torch.LongTensor([maxlen]).to(DEVICE)

                    if sas_model:
                        inter_sas = Interaction({'item_id_list': item_seq, 'item_length': item_len})
                        raw_scores = sas_model.full_sort_predict(inter_sas).detach().cpu().numpy()[0]
                        
                        st.session_state['raw_scores'] = raw_scores
                        st.session_state['has_run'] = True
                        st.session_state['experiment_submitted'] = False

    # ------------------------------------------------------------------
    # 결과 렌더링
    # ------------------------------------------------------------------
    if st.session_state.get('has_run', False) and 'raw_scores' in st.session_state:
        raw_scores = st.session_state['raw_scores']
        
        # --- Logic A: History Boost ---
        scores_A = raw_scores.copy()
        item_counts = {}
        for h in st.session_state['history']:
            raw_id = h['item_id']
            item_counts[raw_id] = item_counts.get(raw_id, 0) + 1
            
        for raw_id, count in item_counts.items():
            if raw_id in token2id:
                idx = token2id[raw_id]
                if idx < len(scores_A):
                    scores_A[idx] += alpha * np.log1p(count)

        topk_A_ids = np.argsort(scores_A)[::-1][:10]

        # --- Logic B: Cycle Filtering ---
        scores_B = scores_A.copy()
        
        for h in st.session_state['history']:
            raw_id = h['item_id']
            days = h['days_ago']
            if raw_id in token2id:
                idx = token2id[raw_id]
                if idx < len(scores_B):
                    c_info = cycle_data.get(raw_id, {})
                    if check_cycle_filtering(days, c_info):
                        scores_B[idx] = -np.inf
        
        topk_B_ids = np.argsort(scores_B)[::-1][:10]

        # --- 매핑 로직 ---
        if 'ab_mapping' not in st.session_state:
            st.session_state['ab_mapping'] = random.choice(['A_is_1', 'B_is_1'])

        mapping = st.session_state['ab_mapping']
        
        if mapping == 'A_is_1':
            opt1_ids, opt1_name = topk_A_ids, "Logic A (구매주기 고려 x)"
            opt2_ids, opt2_name = topk_B_ids, "Logic B (구매주기 고려 o (필터링))"
        else:
            opt1_ids, opt1_name = topk_B_ids, "Logic B (구매주기 고려 o (필터링))"
            opt2_ids, opt2_name = topk_A_ids, "Logic A (구매주기 고려 x)"

        def get_simple_info(idx):
            name, cat = "Unknown", ""
            if idx in id2token:
                raw_id = id2token[idx]
                row = all_df[all_df['item_id'].astype(str) == raw_id]
                if not row.empty:
                    name = row.iloc[0]['Item_Name_KR']
                    cat = row.iloc[0]['L2_KR']
            return f"[{cat}] {name}"

        st.divider()
        st.subheader("⚖️ 블라인드 테스트: 더 만족스러운 추천은?")
        
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("### 🅰️ Option 1")
            for rank, idx in enumerate(opt1_ids):
                if idx == 0: continue
                st.write(f"{rank+1}. {get_simple_info(idx)}")

        with bc2:
            st.markdown("### 🅱️ Option 2")
            for rank, idx in enumerate(opt2_ids):
                if idx == 0: continue
                st.write(f"{rank+1}. {get_simple_info(idx)}")

        st.markdown("---")
        with st.form("ab_test_form"):
            st.write("📝 **평가 입력**")
            choice = st.radio("더 마음에 드는 추천 결과는?", ["Option 1", "Option 2"], horizontal=True)
            reason = st.text_area("이유:")
            
            if st.form_submit_button("제출 및 결과 확인", type="primary"):
                st.session_state['experiment_submitted'] = True
                st.session_state['user_choice'] = choice
                
                log_data = {
                    "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "user_choice": choice,
                    "logic_left": opt1_name,
                    "logic_right": opt2_name,
                    "winner": opt1_name if choice == "Option 1" else opt2_name,
                    "reason": reason
                }
                
                save_df = pd.DataFrame([log_data])
                csv_file = 'ab_test_results.csv'
                try:
                    if not os.path.exists(csv_file):
                        save_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                    else:
                        save_df.to_csv(csv_file, index=False, header=False, mode='a', encoding='utf-8-sig')
                    st.success("데이터 저장 완료!")
                except Exception as e:
                    st.error(f"저장 실패: {e}")

        # ------------------------------------------------------------------
        # 결과 공개 및 색상 강조
        # ------------------------------------------------------------------
        if st.session_state.get('experiment_submitted', False):
            st.divider()
            st.header("🔓 결과 공개")
            
            user_pick = st.session_state['user_choice']
            real_logic = opt1_name if user_pick == "Option 1" else opt2_name
            
            st.success(f"당신의 선택: **{user_pick}**")
            st.info(f"실제 로직: **{real_logic}**")
            
            # 비교를 위한 집합 생성
            set_A = set(topk_A_ids)
            set_B = set(topk_B_ids)

            # [New] 구매 이력 조회용 딕셔너리 생성 (item_id -> 가장 최근 days_ago)
            history_last_days = {}
            for h in st.session_state['history']:
                rid = str(h['item_id'])
                d = h['days_ago']
                # 같은 아이템이 여러 번 있을 경우 가장 최근(작은 숫자) 저장
                if rid not in history_last_days or d < history_last_days[rid]:
                    history_last_days[rid] = d

            rc1, rc2 = st.columns(2)
            
            # zip으로 중복 코드 통합
            for col, ids, name in zip([rc1, rc2], [opt1_ids, opt2_ids], [opt1_name, opt2_name]):
                with col:
                    st.markdown(f"### {name}")
                    for rank, idx in enumerate(ids):
                        if idx == 0: continue
                        
                        info = get_simple_info(idx)
                        
                        # [New] 구매 이력 확인 및 텍스트 추가
                        raw_id = id2token.get(idx, None)
                        # [수정 후: 신규 상품 태그 추가]
                        if raw_id and str(raw_id) in history_last_days:
                            last_day = history_last_days[str(raw_id)]
                            day_str = "오늘" if last_day == 0 else f"{last_day}일 전"
                            info += f" **(↻ {day_str} 구매)**"
                        else:
                        # 구매 기록이 없는 경우
                            info += " **(✨ 신규 추천)**"
                        

                        if name.startswith("Logic A"):
                            # Logic A 목록: Logic B에 없는 아이템 (사라짐) -> 주황색
                            if idx not in set_B:
                                st.markdown(f":orange[{rank+1}. {info}]")
                            else:
                                st.write(f"{rank+1}. {info}")
                        else:
                            # Logic B 목록: Logic A에 없는 아이템 (새로 등장) -> 초록색
                            if idx not in set_A:
                                st.markdown(f":green[{rank+1}. {info}]")
                            else:
                                st.write(f"{rank+1}. {info}")

if __name__ == "__main__":
    main()