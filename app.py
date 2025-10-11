# ====================================================================
#  Streamlit Cloud 環境問題への最終対策コード
# ====================================================================
import subprocess
import sys
import os
import streamlit as st

# Streamlit Cloud上で実行されている場合のみ、ライブラリの強制アップデートを実行
# (ローカル環境では不要なため)
if "STREAMLIT_SHARING_MODE" in os.environ:
    try:
        # pipを使ってgoogle-generativeaiライブラリを最新版に強制的にアップグレード
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "google-generativeai"
        ])
        # 更新が完了したことをトースト通知で知らせる（オプション）
        st.toast("✅ Geminiライブラリを最新版に更新しました。")
    except Exception as e:
        # もし失敗した場合は、エラーを表示してアプリを停止
        st.error(f"ライブラリの強制アップデートに失敗しました: {e}")
        st.stop()
# ====================================================================
#  対策コードここまで
# ====================================================================

## ライブラリインポート
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import json
from io import StringIO
from janome.tokenizer import Tokenizer


# --- 0. Gemini APIキーの設定 ---
# Streamlitのシークレット管理機能からAPIキーを取得
try:
    ## 環境変数からAPIキー読み込み
    gemini_api_key = os.environ["GEMINI_API_B2BAPP"]

    if gemini_api_key:
        genai.configure(api_key = gemini_api_key)
        ## st.success("OK:API kye")
    else:
        st.error("APIキーが正しく設定されていません。")
        st.stop("APIキーエラーのため処理を停止します。")
except Exception as e:
    st.error(f"APIキー設定に予期せぬエラーが発生しました:{e}")
    exit()

# --- 1. Janomeの初期化 ---
try:
    janome_tokenizer = Tokenizer()
except Exception as e:
    st.error(f"Janomeの初期化に失敗しました。エラー: {e}")
    st.stop()


# --- 2. Big5性格分析 + ACN独自性格のロジック ---
PERSONALITY_WORDS = {
    ## Big5
    '外向性': ['みんな', '楽しい', 'パーティー', '会う', '話す', '最高', '！', '（笑）'],
    '協調性': ['協力', '一緒', '手伝う', 'ありがとう', 'お願いします', '感謝', '私たち'],
    '誠実性': ['計画', '確実', 'べき', '重要', '責任', 'しっかり', '管理', '報告'],
    '神経症傾向': ['心配', '不安', '問題', '難しい', '大変', 'リスク', 'すみません'],
    '開放性': ['新しい', 'アイデア', '面白い', '試す', '想像', 'わくわく', '？', 'なるほど'],
    ## ACNオリジナル
    '短気': ['まだ', '遅い', 'おそい', 'いそげ', '急げ', 'いそぐ', '急ぐ', 'やば', '早く'],
    'パワハラ': ['おいおい', 'まじかよ', 'やれ', 'ふざけるな']
}

## チャットのテキストをJanomeで単語に切り分け
def analyze_personality(text):
    if not isinstance(text, str) or not text.strip():
        return {p: 0 for p in PERSONALITY_WORDS.keys()}
    # ↓ Janomeの処理に置き換え
    words = [token.surface for token in janome_tokenizer.tokenize(text)]
    scores = {p: 0 for p in PERSONALITY_WORDS.keys()}
    for personality, keywords in PERSONALITY_WORDS.items():
        for word in keywords:
            scores[personality] += words.count(word)
    return scores

## 文章中のキーワード数をカウントし性格タイプごとのスコア算出
def get_dominant_personality(scores):
    if not scores or sum(scores.values()) == 0:
        return "分析不能"
    return max(scores, key=scores.get)

## 性格の類似度を計算する関数
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

## 性格スコアを元にレーダーチャートを生成
def create_multi_user_radar_chart(result_df, my_name):
    labels = list(PERSONALITY_WORDS.keys())
    fig = go.Figure()

    # 自分以外のメンバーに使用する色のリストを定義
    other_colors = [
        '#1f77b4',  # 青
        '#2ca02c',  # 緑
        '#9467bd',  # 紫
        '#ff7f0e',  # オレンジ
        '#8c564b',  # 茶色
        '#e377c2',  # ピンク
        '#7f7f7f',  # グレー
        '#bcbd22',  # オリーブ
        '#17becf'   # シアン
    ]
    color_index = 0

    score_details_df = result_df['スコア詳細'].apply(pd.Series)
    max_score = score_details_df.max().max() if not score_details_df.empty else 1

    for index, row in result_df.iterrows():
        user_name = row['ユーザー']
        scores = row['スコア詳細']
        values = list(scores.values())

        # ユーザー名に応じて色を決定
        if user_name == my_name:
            line_color = 'red'
        else:
            line_color = other_colors[color_index % len(other_colors)]
            color_index += 1
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=user_name,
            # line=dict(color=...) で線の色を指定
            line=dict(color=line_color),
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

# --- 3. Gemini APIを用いた関数群 ---
def generate_persona_with_retry(target_user_name, target_user_scores, target_user_text, my_scores, max_retries=3):
    """
    Gemini APIを呼び出してペルソナを生成する関数
    """
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    default_response = {
        "persona": "AIによる人柄の解析に失敗しました。",
        "common_point": "共通点の解析に失敗しました。",
        "communication_point": "コミュニケーションのポイントの解析に失敗しました。"
    }
    prompt = f"""
    あなたは、優秀な組織人事コンサルタントです。以下の情報を基に、{target_user_name}さんのペルソナを分析してください。
    # 分析対象者の情報
    - 氏名: {target_user_name}
    - 性格スコア: {target_user_scores}
    - チャット発言の抜粋: {target_user_text[:200]}
    # 比較対象者（自分）の情報
    - 性格スコア: {my_scores}
    #【最重要ルール】
    - **必ず、以下のキーを持つJSONオブジェクト"だけ"を生成してください。**
    - **解説や前置き、```jsonのような追加の文字列は絶対に含めないでください。**
    {{
    "persona": "（{target_user_name}さんの人柄を50文字程度で説明）",
    "common_point": "（自分との性格スコアの共通点を30文字程度で説明）",
    "communication_point": "（円滑な関係を築くためのポイントを50文字程度で説明）"
    }}
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            persona_data = json.loads(cleaned_text)
            if all(k in persona_data for k in ["persona", "common_point", "communication_point"]):
                return persona_data
        except (json.JSONDecodeError, Exception) as e:
            st.error(f"AIからの応答処理中にエラーが発生しました（試行 {attempt + 1}回目）: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                st.text_area("AIからの生の応答:", response.text, height=150)
            if attempt == max_retries - 1:
                return default_response
    return default_response

def generate_team_atmosphere(text, max_retries=3):
    """
    MTGの会話テキストからチームの雰囲気を分析する関数
    """
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    default_response = {"atmosphere": "分析失敗", "description": "AIによる分析に失敗しました。", "weather": "霧"}
    weather_options = ['快晴', '晴れ', '薄曇り', '曇り', '雨', '雪', '雷', '霧', '暴風']
    prompt = f"""
    あなたは、経験豊富な組織開発コンサルタントです。以下のオンラインミーティングの会話テキスト全体から、チーム全体の雰囲気を分析してください。
    # 分析対象の会話テキスト（抜粋）
    {text[:4000]}
    #【最重要ルール】
    - **必ず、以下のキーを持つJSONオブジェクト"だけ"を生成してください。**
    - **解説や前置き、```jsonのような追加の文字列は絶対に含めないでください。**
    - **"weather"の値は、必ず以下のリストから最も適切だと思うものを1つだけ選んでください。**
      {weather_options}
    {{
      "atmosphere": "（チームの雰囲気を15文字程度で表現）",
      "description": "（なぜその雰囲気だと判断したか、理由を30文字程度で説明）",
      "weather": "（上記のリストから選択した天気）"
    }}
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            atmosphere_data = json.loads(cleaned_text)
            if all(k in atmosphere_data for k in ["atmosphere", "description", "weather"]):
                return atmosphere_data
        except (json.JSONDecodeError, Exception) as e:
            st.error(f"チーム雰囲気のAI分析中にエラーが発生しました（試行 {attempt + 1}回目）: {e}")
            if attempt == max_retries - 1:
                return default_response
    return default_response

def get_weather_icon(weather_str):
    """ 天気の文字列に対応する絵文字を返す """
    weather_map = {
        '快晴': '☀️', '晴れ': '晴️', '薄曇り': '🌥️', '曇り': '☁️', '雨': '🌧️',
        '雪': '❄️', '雷': '⚡️', '霧': '🌫️', '暴風': '🌪️'
    }
    return weather_map.get(weather_str, '❓')

##ここから修正
# --- 3-3. Gemini APIを用いた総合評価関数 ---
def generate_overall_evaluation(atmosphere_result, result_df, my_name, max_retries=3):
    """
    全ての分析結果を統合し、プロジェクトへの参加推奨度を評価する関数
    """
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    # AIに渡すために、これまでの分析結果を要約します
    team_atmosphere = atmosphere_result.get('atmosphere', '不明')
    team_weather = atmosphere_result.get('weather', '不明')
    
    my_data = result_df[result_df['ユーザー'] == my_name]
    my_dominant_personality = my_data['最も強い性格傾向'].iloc[0] if not my_data.empty else '不明'
    
    other_members_data = result_df[result_df['ユーザー'] != my_name]
    # `mean()`で平均値を計算。データがない場合は0とします。
    average_match_score = other_members_data['自分との性格マッチ度 (%)'].mean() if not other_members_data.empty else 0
    # `value_counts()`で性格タイプの人数を数え、`to_dict()`で辞書に変換します
    team_composition = other_members_data['最も強い性格傾向'].value_counts().to_dict()

    # 解析失敗時のデフォルト応答
    default_response = {
        "recommendation": "自己判断に委ねる",
        "reason": "AIによる総合評価に失敗しました。"
    }

    # AIに選ばせる選択肢を定義
    recommendation_options = ["強く推奨する", "推奨する", "自己判断に委ねる", "推奨しない"]

    prompt = f"""
    あなたは、超一流の組織人事コンサルタント兼キャリアアドバイザーです。
    以下の多角的な分析結果を基に、ユーザー（{my_name}さん）がこのプロジェクトに参加すべきかどうか、総合的な評価とアドバイスをしてください。

    # 分析データ
    1. **チーム全体の雰囲気**:
       - 雰囲気: {team_atmosphere} (天気で言うと「{team_weather}」)

    2. **ユーザーとチームメンバーの性格分析**:
       - ユーザーの最も強い性格傾向: {my_dominant_personality}
       - チームメンバーとの平均性格マッチ度: {average_match_score:.1f}%
       - チームメンバーの性格傾向の内訳: {team_composition}

    #【最重要ルール】
    - **必ず、以下のキーを持つJSONオブジェクト"だけ"を生成してください。**
    - **解説や前置き、```jsonのような追加の文字列は絶対に含めないでください。**
    - **"recommendation"の値は、必ず以下のリストから最も適切だと思うものを1つだけ選んでください。**
      {recommendation_options}
    - **"reason"は、その結論に至った根拠を20文字程度で簡潔に述べてください。**

    {{
      "recommendation": "（上記のリストから選択）",
      "reason": "（評価理由を20文字程度で説明）"
    }}
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            evaluation_data = json.loads(cleaned_text)
            
            if all(k in evaluation_data for k in ["recommendation", "reason"]):
                return evaluation_data # 成功
        except (json.JSONDecodeError, Exception) as e:
            st.error(f"総合評価のAI分析中にエラーが発生しました（試行 {attempt + 1}回目）: {e}")
            if attempt == max_retries - 1:
                return default_response
    
    return default_response

def get_recommendation_color(recommendation_str):
    """ 推奨度に応じて色を返す """
    if recommendation_str == "強く推奨する":
        return "green"
    elif recommendation_str == "推奨する":
        return "blue"
    elif recommendation_str == "推奨しない":
        return "red"
    else: # 自己判断に委ねる
        return "orange"
##修正ここまで

# --- 4. Streamlitアプリケーションの画面 ---
st.title('アサイン検討PJ 性格分析アプリ 💬')
st.write('アサイン予定のPJメンバーのチャットデータ、MTG会話データとあなたのチャットデータ(CSV)をアップロードすると、チームの雰囲気やメンバーの性格傾向を分析し、PJとあなたのマッチングを診断します。')
st.write('---')

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("👥PJメンバーのTeamsチャット")
    chat_files = st.file_uploader("PJのチャットCSVを選択", type="csv", accept_multiple_files=True, key="chat_uploader")
with col2:
    st.subheader("👥PJメンバーのMTG会話")
    transcript_files = st.file_uploader("音声からテキスト変換したCSVを選択", type="csv", accept_multiple_files=True, key="transcript_uploader")
with col3:
    st.subheader("👤自分のTeamsチャット")
    my_file = st.file_uploader("自分のチャットのCSVを選択", type="csv", accept_multiple_files=False, key="mychat_uploader")
st.write('---')

if (chat_files or transcript_files) and my_file:
    try:
        # --- ファイルの読み込み処理 ---
        team_chat_dfs = []
        if chat_files:
            for file in chat_files:
                file.seek(0)
                try: df_single = pd.read_csv(file, encoding='shift_jis')
                except UnicodeDecodeError:
                    file.seek(0)
                    df_single = pd.read_csv(file, encoding='utf-8')
                team_chat_dfs.append(df_single)
        team_chat_df = pd.concat(team_chat_dfs, ignore_index=True) if team_chat_dfs else pd.DataFrame()

        transcript_text = ""
        if transcript_files:
            team_transcript_dfs = []
            for file in transcript_files:
                file.seek(0)
                try: df_single = pd.read_csv(file, encoding='shift_jis')
                except UnicodeDecodeError:
                    file.seek(0)
                    df_single = pd.read_csv(file, encoding='utf-8')
                team_transcript_dfs.append(df_single)
            if team_transcript_dfs:
                team_transcript_df = pd.concat(team_transcript_dfs, ignore_index=True)
                if 'message' in team_transcript_df.columns:
                    transcript_text = ' '.join(team_transcript_df['message'].fillna('').astype(str))

        try:
            my_file.seek(0)
            my_df = pd.read_csv(my_file, encoding='shift_jis')
        except UnicodeDecodeError:
            my_file.seek(0)
            my_df = pd.read_csv(my_file, encoding='utf-8')

        if 'user' in my_df.columns and not my_df.empty:
            my_name = my_df['user'].iloc[0]
            st.info(f"あなたのユーザー名を「{my_name}」として認識しました。")
        else:
            st.error("あなたのCSVファイルに 'user' 列が存在しないか、データが空です。"); st.stop()
        
        df = pd.concat([team_chat_df, my_df], ignore_index=True)
        st.success(f'{len(chat_files) + len(transcript_files) + 1}個のファイルの読み込みに成功！')

        if st.button('分析を実行する'):
            st.write('---'); st.header('分析結果')
            atmosphere_result, result_df = None, pd.DataFrame() # 結果を保存する変数を初期化

            # --- チームの雰囲気分析 ---
            if transcript_text:
                st.subheader('🗣️ PJチームの雰囲気')
                with st.spinner('AIがチームの雰囲気を分析中です...'):
                    atmosphere_result = generate_team_atmosphere(transcript_text)
                    weather_str = atmosphere_result.get('weather', '霧')
                    col_atm1, col_atm2, col_atm3 = st.columns(3)
                    col_atm1.metric("現在のチームの天気", weather_str, get_weather_icon(weather_str))
                    col_atm2.info(f"**雰囲気**: {atmosphere_result.get('atmosphere', 'N/A')}")
                    col_atm3.success(f"**理由**: {atmosphere_result.get('description', 'N/A')}")
                st.write('---')
            
            # --- 性格マッチ度分析 ---
            if not df.empty and 'user' in df.columns and df['user'].nunique() > 1:
                df['message'] = df['message'].fillna('')
                user_texts = df.groupby('user')['message'].agg(' '.join).reset_index()
                results = []
                with st.spinner('性格スコアを分析中です...'):
                    for _, row in user_texts.iterrows():
                        scores = analyze_personality(row['message'])
                        results.append({
                            'ユーザー': row['user'], '最も強い性格傾向': get_dominant_personality(scores),
                            '発言数': df[df['user'] == row['user']].shape[0],
                            'スコア詳細': scores, '全発言': row['message']
                        })
                result_df = pd.DataFrame(results)
                if my_name in result_df['ユーザー'].values:
                    my_scores_row = result_df[result_df['ユーザー'] == my_name].iloc[0]
                    my_scores = my_scores_row['スコア詳細']
                    my_scores_list = list(my_scores.values())
                    match_percentages = [
                        round(calculate_cosine_similarity(my_scores_list, list(row['スコア詳細'].values())) * 100, 2)
                        if row['ユーザー'] != my_name else 100.0
                        for _, row in result_df.iterrows()
                    ]
                    result_df['自分との性格マッチ度 (%)'] = match_percentages
                    st.subheader(f'👤 {my_name} とPJメンバーの性格マッチ度')
                    st.dataframe(result_df[['ユーザー', '最も強い性格傾向', '発言数', '自分との性格マッチ度 (%)']].sort_values('自分との性格マッチ度 (%)', ascending=False))
                    st.write('---')
                    st.subheader('📈 自分とPJメンバーの性格比較チャート')
                    st.plotly_chart(create_multi_user_radar_chart(result_df, my_name), use_container_width=True)
                    st.write('---')
                    st.subheader('🤖 PJメンバーのペルソナ分析')
                    other_users_df = result_df[result_df['ユーザー'] != my_name].sort_values('自分との性格マッチ度 (%)', ascending=False)
                    with st.spinner('AIが各メンバーのペルソナを生成中です...'):
                        user_list = list(other_users_df.iterrows())
                        for i in range(0, len(user_list), 2):
                            col1, col2 = st.columns(2)
                            # 1人目
                            _, row1 = user_list[i]
                            with col1.expander(f"**{row1['ユーザー']}さん** のペルソナを見る", expanded=True):
                                persona = generate_persona_with_retry(row1['ユーザー'], row1['スコア詳細'], row1['全発言'], my_scores)
                                st.info(f"**人柄**: {persona.get('persona', 'N/A')}")
                                st.success(f"**共通点**: {persona.get('common_point', 'N/A')}")
                                st.warning(f"**ポイント**: {persona.get('communication_point', 'N/A')}")
                            # 2人目 (存在すれば)
                            if (i + 1) < len(user_list):
                                _, row2 = user_list[i + 1]
                                with col2.expander(f"**{row2['ユーザー']}さん** のペルソナを見る", expanded=True):
                                    persona = generate_persona_with_retry(row2['ユーザー'], row2['スコア詳細'], row2['全発言'], my_scores)
                                    st.info(f"**人柄**: {persona.get('persona', 'N/A')}")
                                    st.success(f"**共通点**: {persona.get('common_point', 'N/A')}")
                                    st.warning(f"**ポイント**: {persona.get('communication_point', 'N/A')}")
            else:
                 st.info("性格分析を行うには、PJメンバーのチャットファイルと自分のチャットファイルを両方アップロードしてください。")

            ##ここから修正
            # --- 総合評価 ---
            st.write('---')
            st.header('📊 総合評価')

            # 雰囲気分析と性格分析の両方のデータが揃っているか確認
            if atmosphere_result and not result_df.empty:
                with st.spinner('AIがすべての結果を統合し、最終評価を生成中です...'):
                    evaluation = generate_overall_evaluation(atmosphere_result, result_df, my_name)
                    recommendation = evaluation.get('recommendation', '評価不能')
                    reason = evaluation.get('reason', '理由の取得に失敗しました。')
                    color = get_recommendation_color(recommendation)
                    
                    st.markdown(f"### あなたへの推奨度: <font color='{color}'>**{recommendation}**</font>", unsafe_allow_html=True)
                    st.info(f"**理由**: {reason}")
            else:
                st.warning("総合評価を行うには、PJメンバーの「MTG会話」と「Teamsチャット」の両方のデータをアップロードして分析を実行する必要があります。")
            ##修正ここまで

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")