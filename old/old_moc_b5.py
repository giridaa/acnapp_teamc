import streamlit as st
import pandas as pd
import MeCab
import re
import plotly.graph_objects as go

# --- 1. MeCabの初期化 ---
MECABRC_PATH = "C:/Program Files/MeCab/etc/mecabrc"

try:
    mecab = MeCab.Tagger(f'-r "{MECABRC_PATH}" -Owakati')
except Exception as e:
    st.error(f"MeCabの初期化に失敗しました。MeCab本体のインストールと環境パス設定を確認してください。エラー: {e}")
    st.stop()


# --- 2. Big5性格分析のロジック---
PERSONALITY_WORDS = {
    '外向性': ['みんな', '楽しい', 'パーティー', '会う', '話す', '最高', '！'],
    '協調性': ['協力', '一緒', '手伝う', 'ありがとう', 'お願いします', '感謝'],
    '誠実性': ['計画', '確実', 'べき', '重要', '責任', 'しっかり', '管理'],
    '神経症傾向': ['心配', '不安', '問題', '難しい', '大変', 'リスク'],
    '開放性': ['新しい', 'アイデア', '面白い', '試す', '想像', 'わくわく', '？']
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

# --- ★★★★★ ここから変更 ★★★★★ ---
def create_multi_user_radar_chart(result_df):
    """
    複数ユーザーの結果データフレームからPlotlyのレーダーチャートを作成する関数
    """
    # グラフの項目（ラベル）を取得
    labels = list(PERSONALITY_WORDS.keys())
    
    # グラフオブジェクトの作成
    fig = go.Figure()

    # 全ユーザーのスコアからチャートの最大値を計算（全ユーザーでスケールを統一するため）
    score_details_df = result_df['スコア詳細'].apply(pd.Series)
    max_score = score_details_df.max().max()

    # ユーザーごとにレーダーチャートのトレースを追加
    for index, row in result_df.iterrows():
        user_name = row['ユーザー']
        scores = row['スコア詳細']
        values = list(scores.values())

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=user_name,
            hovertemplate=f'<b>{user_name}</b><br>%{{theta}}: %{{r}}<extra></extra>' # マウスオーバー時の表示を調整
        ))

    # グラフのレイアウトを更新
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_score if max_score > 0 else 1] # 全員の最大値に合わせる
            )),
        showlegend=True,
        title="全ユーザーの性格スコア比較"
    )

    return fig
# --- ★★★★★ ここまで変更 ★★★★★ ---


# --- 3. Streamlitアプリケーションの画面 ---
st.title('Teamsチャット 性格分析アプリ 💬')
st.write('CSVファイルをアップロードすると、発言者ごとの性格を簡易的に分析します。')

uploaded_file = st.file_uploader("チャットデータ（CSVファイル）をアップロードしてください", type="csv")

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='shift_jis')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8')

        if 'user' not in df.columns or 'message' not in df.columns:
            st.error("エラー: CSVファイルには 'user' と 'message' の列が必要です。")
        else:
            st.success('ファイルの読み込みに成功しました！')
            st.dataframe(df.head())

            if st.button('性格を分析する'):
                st.write('---')
                st.write('**分析結果**')
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
                st.dataframe(result_df[['ユーザー', '最も強い性格傾向', '発言数']])

                with st.expander("各ユーザーのスコア詳細を見る"):
                    score_details_df = result_df.set_index('ユーザー')['スコア詳細'].apply(pd.Series)
                    st.dataframe(score_details_df)
                
                # --- ★★★★★ ここから変更 ★★★★★ ---
                st.write('---')
                st.write('**📈 全ユーザーの性格スコア比較チャート**')
                
                if not result_df.empty:
                    # 複数ユーザー対応の関数を呼び出す
                    fig = create_multi_user_radar_chart(result_df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("分析対象のユーザーが見つかりませんでした。")
                # --- ★★★★★ ここまで変更 ★★★★★ ---

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")