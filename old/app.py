# app.py

import streamlit as st
import pandas as pd
import MeCab
import numpy as np
import plotly.graph_objects as go
from io import StringIO

# --- 1. MeCabの初期化 (Google Colab向け) ---
try:
    mecab = MeCab.Tagger("-Owakati")
except Exception as e:
    st.error(f"MeCabの初期化に失敗しました。エラー: {e}")
    st.info("Google Colab環境では、!pip install mecab-python3 unidic-lite を実行してください。")
    st.stop()


# --- 2. Big5性格分析のロジック ---
PERSONALITY_WORDS = {
    '外向性': ['みんな', '楽しい', 'パーティー', '会う', '話す', '最高', '！', '（笑）'],
    '協調性': ['協力', '一緒', '手伝う', 'ありがとう', 'お願いします', '感謝', '私たち'],
    '誠実性': ['計画', '確実', 'べき', '重要', '責任', 'しっかり', '管理', '報告'],
    '神経症傾向': ['心配', '不安', '問題', '難しい', '大変', 'リスク', 'すみません'],
    '開放性': ['新しい', 'アイデア', '面白い', '試す', '想像', 'わくわく', '？', 'なるほど']
}

def analyze_personality(text):
    if not isinstance(text, str) or not text.strip():
        return {p: 0 for p in PERSONALITY_WORDS.keys()}
    words = mecab.parse(text).split()
    scores = {p: 0 for p in PERSONALITY_WORDS.keys()}
    for personality, keywords in PERSONALITY_WORDS.items():
        for word in keywords:
            scores[personality] += words.count(word)
    return scores

def get_dominant_personality(scores):
    if not scores or sum(scores.values()) == 0:
        return "分析不能"
    return max(scores, key=scores.get)

def calculate_cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def create_multi_user_radar_chart(result_df):
    labels = list(PERSONALITY_WORDS.keys())
    fig = go.Figure()
    score_details_df = result_df['スコア詳細'].apply(pd.Series)
    max_score = score_details_df.max().max()
    for index, row in result_df.iterrows():
        user_name = row['ユーザー']
        scores = row['スコア詳細']
        values = list(scores.values())
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=user_name,
            hovertemplate=f'<b>{user_name}</b><br>%{{theta}}: %{{r}}<extra></extra>'
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_score if max_score > 0 else 1]
            )),
        showlegend=True,
        title="全ユーザーの性格スコア比較"
    )
    return fig


# --- 3. Streamlitアプリケーションの画面 ---
st.title('Teamsチャット 性格分析アプリ 💬')
st.write('アサイン予定のPJメンバーとあなたのチャットデータ(CSV)をアップロードすると、性格傾向を分析し、PJメンバーとの「性格マッチ度」を診断します。')
st.write('---')

# 2つのカラムを作成
col1, col2 = st.columns(2)

# 左側のカラムにチームメンバー用のアップローダーを配置
with col1:
    st.subheader("PJメンバーのチャット")
    team_files = st.file_uploader(
        "CSVファイルを複数選択してください",
        type="csv",
        accept_multiple_files=True,
        key="team_uploader"
    )

# 右側のカラムに自分用のアップローダーを配置
with col2:
    st.subheader("自分のチャット")
    my_file = st.file_uploader(
        "CSVファイルを1つ選択してください",
        type="csv",
        accept_multiple_files=False,
        key="my_uploader"
    )

st.write('---')

if team_files and my_file:
    try:
        # --- チームメンバーのデータ読み込み ---
        team_dfs = []
        for file in team_files:
            file.seek(0)
            try:
                df_single = pd.read_csv(file, encoding='shift_jis')
            except UnicodeDecodeError:
                file.seek(0)
                df_single = pd.read_csv(file, encoding='utf-8')
            team_dfs.append(df_single)
        team_df = pd.concat(team_dfs, ignore_index=True)

        # --- あなたのデータ読み込み ---
        my_file.seek(0)
        try:
            my_df = pd.read_csv(my_file, encoding='shift_jis')
        except UnicodeDecodeError:
            my_file.seek(0)
            my_df = pd.read_csv(my_file, encoding='utf-8')

        # --- あなたのユーザー名を取得 ---
        my_name = ""
        if 'user' in my_df.columns and not my_df.empty:
            my_name = my_df['user'].iloc[0]
            st.info(f"あなたのユーザー名を「{my_name}」として認識しました。")
        else:
            st.error("あなたのCSVファイルに 'user' 列が存在しないか、データが空です。")
            st.stop()
        
        # --- 全データを結合 ---
        df = pd.concat([team_df, my_df], ignore_index=True)
        
        if 'user' not in df.columns or 'message' not in df.columns:
            st.error("エラー: CSVファイルには 'user' と 'message' の列が必要です。")
        else:
            st.success(f'{len(team_files) + 1}個のファイルの読み込みに成功しました！')
            
            with st.expander("読み込んだデータを確認する"):
                st.write("▼ チームメンバーのデータ（先頭5行）")
                st.dataframe(team_df.head())
                st.write("▼ あなたのデータ（先頭5行）")
                st.dataframe(my_df.head())

            if st.button('PJメンバーとの性格マッチングを確認'):
                st.write('---')
                st.header('分析結果')
                df['message'] = df['message'].fillna('')
                user_texts = df.groupby('user')['message'].apply(' '.join).reset_index()
                
                results = []
                with st.spinner('分析中です...'):
                    for index, row in user_texts.iterrows():
                        user = row['user']
                        text = row['message']
                        scores = analyze_personality(text)
                        dominant_personality = get_dominant_personality(scores)
                        results.append({
                            'ユーザー': user,
                            '最も強い性格傾向': dominant_personality,
                            '発言数': df[df['user'] == user].shape[0],
                            'スコア詳細': scores
                        })
                
                result_df = pd.DataFrame(results)

                if my_name in result_df['ユーザー'].values:
                    my_scores_row = result_df[result_df['ユーザー'] == my_name].iloc[0]
                    my_scores_list = list(my_scores_row['スコア詳細'].values())
                    match_percentages = []
                    for index, row in result_df.iterrows():
                        if row['ユーザー'] == my_name:
                            match_percentages.append(100.0)
                            continue
                        other_scores_list = list(row['スコア詳細'].values())
                        similarity = calculate_cosine_similarity(my_scores_list, other_scores_list)
                        match_percentages.append(round(similarity * 100, 2))
                    
                    result_df['自分との性格マッチ度 (%)'] = match_percentages
                    
                    st.subheader(f'👤 {my_name} とPJメンバーの性格マッチ度')
                    st.dataframe(
                        result_df[['ユーザー', '最も強い性格傾向', '発言数', '自分との性格マッチ度 (%)']]
                        .sort_values('自分との性格マッチ度 (%)', ascending=False)
                    )
                else:
                    st.warning(f"'{my_name}'のデータが分析結果に含まれていませんでした。")
                    st.dataframe(result_df[['ユーザー', '最も強い性格傾向', '発言数']])
                
                st.write('---')
                st.subheader('📈 全ユーザーの性格スコア比較チャート')
                if not result_df.empty:
                    fig = create_multi_user_radar_chart(result_df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("分析対象のユーザーが見つかりませんでした。")
                
                with st.expander("各ユーザーのスコア詳細を見る"):
                    score_details_df = result_df.set_index('ユーザー')['スコア詳細'].apply(pd.Series)
                    st.dataframe(score_details_df)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")