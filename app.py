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

# --- 3. Gemini APIを用いたペルソナ生成関数 ---
def generate_persona_with_retry(target_user_name, target_user_scores, target_user_text, my_scores, max_retries=3):
    """
    Gemini APIを呼び出してペルソナを生成する関数（リトライ機能とJSONパース機能付き）
    """
    ## Geminiのモデルを指定
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    ## 解析に失敗した場合に表示する内容をあらかじめ定義
    default_response = {
        "persona": "AIによる人柄の解析に失敗しました。もう一度お試しください。",
        "common_point": "共通点の解析に失敗しました。",
        "communication_point": "コミュニケーションのポイントの解析に失敗しました。"
    }

    prompt = f"""
    あなたは、優秀な組織人事コンサルタントです。
    以下の情報を基に、{target_user_name}さんのペルソナを分析してください。

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
    ## max_retriesで指定した回数だけ、成功するまで処理を試み
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            ## Geminiが出力することがある余計な文字列を削除
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            ## AIの回答（文字列）をJSON形式（辞書）に変換
            persona_data = json.loads(cleaned_text)
            
            ## 必要な項目がすべて含まれているかチェック
            if all(k in persona_data for k in ["persona", "common_point", "communication_point"]):
                return persona_data  ## 成功したら、結果の辞書を返す
                
        except (json.JSONDecodeError, Exception) as e:
            st.error(f"AIからの応答処理中にエラーが発生しました（試行 {attempt + 1}回目）")
            st.error(f"エラー内容: {e}")
            # response変数が存在し、text属性を持つか確認
            if 'response' in locals() and hasattr(response, 'text'):
                st.text_area("AIからの生の応答:", response.text, height=150)
            
            if attempt == max_retries - 1:
                st.warning(f"AIによるペルソナ生成に失敗しました（試行回数: {max_retries}回）。")
                return default_response
    
    return default_response

##ここから修正
# --- 3-2. Gemini APIを用いたチーム雰囲気分析関数 ---
def generate_team_atmosphere(text, max_retries=3):
    """
    MTGの会話テキストからチームの雰囲気を分析する関数
    """
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    # 解析失敗時のデフォルト応答
    default_response = {
        "atmosphere": "分析失敗",
        "description": "AIによる分析に失敗しました。テキストが短すぎるか、内容が不適切でないか確認してください。",
        "weather": "霧"
    }

    # 天気の選択肢を定義
    weather_options = ['快晴', '晴れ', '薄曇り', '曇り', '雨', '雪', '雷', '霧', '暴風']

    prompt = f"""
    あなたは、経験豊富な組織開発コンサルタントです。
    以下のオンラインミーティングの会話テキスト全体から、チーム全体の雰囲気を分析してください。

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
                return atmosphere_data # 成功
        except (json.JSONDecodeError, Exception) as e:
            st.error(f"チーム雰囲気のAI分析中にエラーが発生しました（試行 {attempt + 1}回目）: {e}")
            if attempt == max_retries - 1:
                st.warning(f"AIによるチーム雰囲気の分析に失敗しました。")
                return default_response
    
    return default_response

def get_weather_icon(weather_str):
    """ 天気の文字列に対応する絵文字を返す """
    weather_map = {
        '快晴': '☀️', '晴れ': '晴️', '薄曇り': '🌥️', '曇り': '☁️',
        '雨': '🌧️', '雪': '❄️', '雷': '⚡️', '霧': '🌫️', '暴風': '🌪️'
    }
    return weather_map.get(weather_str, '❓') # 見つからない場合は「?」を返す
##修正ここまで

# --- 4. Streamlitアプリケーションの画面 ---
st.title('アサイン検討PJ 性格分析アプリ 💬')
st.write('アサイン予定のPJメンバーのチャットデータ、MTG会話データとあなたのチャットデータ(CSV)をアップロードすると、性格傾向を分析し、PJメンバーとの「性格マッチ度」を診断します。')
st.write('---')

col1, col2, col3 = st.columns(3)

## CSVアップローダー
with col1:
    st.subheader("👥PJメンバーのTeamsチャット")
    # アップロードされたファイルを受け取る変数を変更
    chat_files = st.file_uploader(
        "PJのチャットCSVを選択してください",
        type="csv",
        accept_multiple_files=True,
        key="chat_uploader"
    )

with col2:
    st.subheader("👥PJメンバーのMTG会話")
    # アップロードされたファイルを受け取る変数を変更
    transcript_files = st.file_uploader(
        "音声からテキスト変換したCSVを選択してください",
        type="csv",
        accept_multiple_files=True,
        key="transcript_uploader"
    )

with col3:
    st.subheader("👤自分のTeamsチャット")
    my_file = st.file_uploader(
        "自分のチャットのCSVを選択してください",
        type="csv",
        accept_multiple_files=False,
        key="mychat_uploader"
    )

st.write('---')

# チームメンバーのファイル（チャット or MTG）がどちらか一方でもアップロードされたら処理に進むように条件を変更
if (chat_files or transcript_files) and my_file:
    try:
##ここから修正
        # --- ファイルの読み込み処理 ---
        # 性格分析用のチャットデータを読み込む
        team_chat_dfs = []
        if chat_files:
            for file in chat_files:
                file.seek(0)
                try:
                    df_single = pd.read_csv(file, encoding='shift_jis')
                except UnicodeDecodeError:
                    file.seek(0)
                    df_single = pd.read_csv(file, encoding='utf-8')
                team_chat_dfs.append(df_single)
        
        team_chat_df = pd.DataFrame()
        if team_chat_dfs:
            team_chat_df = pd.concat(team_chat_dfs, ignore_index=True)

        # チーム雰囲気分析用のMTG会話データを読み込む
        team_transcript_dfs = []
        transcript_text = ""
        if transcript_files:
            for file in transcript_files:
                file.seek(0)
                try:
                    df_single = pd.read_csv(file, encoding='shift_jis')
                except UnicodeDecodeError:
                    file.seek(0)
                    df_single = pd.read_csv(file, encoding='utf-8')
                team_transcript_dfs.append(df_single)
        
        team_transcript_df = pd.DataFrame()
        if team_transcript_dfs:
            team_transcript_df = pd.concat(team_transcript_dfs, ignore_index=True)
            if 'message' in team_transcript_df.columns:
                # 全ての発言を一つのテキストに結合
                transcript_text = ' '.join(team_transcript_df['message'].fillna('').astype(str))

        # 自分のデータを読み込む
        my_file.seek(0)
        try:
            my_df = pd.read_csv(my_file, encoding='shift_jis')
        except UnicodeDecodeError:
            my_file.seek(0)
            my_df = pd.read_csv(my_file, encoding='utf-8')
##修正ここまで

        my_name = ""
        if 'user' in my_df.columns and not my_df.empty:
            my_name = my_df['user'].iloc[0]
            st.info(f"あなたのユーザー名を「{my_name}」として認識しました。")
        else:
            st.error("あなたのCSVファイルに 'user' 列が存在しないか、データが空です。")
            st.stop()
        
        # 性格分析にはチャットデータと自分のデータを結合
        df = pd.concat([team_chat_df, my_df], ignore_index=True)
        
        if 'user' not in df.columns or 'message' not in df.columns:
            # チャットデータがない場合は警告にとどめる
            if not chat_files:
                 st.warning("性格分析の対象となるPJメンバーのチャットファイルがアップロードされていません。")
            else:
                 st.error("エラー: チャットCSVファイルには 'user' と 'message' の列が必要です。")
        else:
            st.success(f'{len(chat_files) + len(transcript_files) + 1}個のファイルの読み込みに成功しました！')
            
            with st.expander("読み込んだデータを確認する"):
                if not team_chat_df.empty:
                    st.write("▼ チームメンバーのチャットデータ（先頭5行）")
                    st.dataframe(team_chat_df.head())
                if not team_transcript_df.empty:
                    st.write("▼ チームメンバーのMTG会話データ（先頭5行）")
                    st.dataframe(team_transcript_df.head())
                st.write("▼ あなたのデータ（先頭5行）")
                st.dataframe(my_df.head())

        ## ボタンを押して性格マッチ度分析を実行
        if st.button('分析を実行する'):
            st.write('---')
            st.header('分析結果')

##ここから修正
            # --- チームの雰囲気分析（MTGデータがある場合のみ実行）---
            if transcript_text:
                st.subheader('🗣️ MTGの会話から分析したチームの雰囲気')
                with st.spinner('AIがチームの雰囲気を分析中です...'):
                    atmosphere_result = generate_team_atmosphere(transcript_text)
                    
                    # 天気をアイコンで表示
                    weather_str = atmosphere_result.get('weather', '霧')
                    weather_icon = get_weather_icon(weather_str)
                    
                    # 結果を3カラムで表示
                    col_atm1, col_atm2, col_atm3 = st.columns(3)
                    with col_atm1:
                        st.metric(label="現在のチームの天気", value=weather_str, delta=weather_icon)
                    with col_atm2:
                        st.markdown("**チームの雰囲気**")
                        st.info(f"{atmosphere_result.get('atmosphere', 'N/A')}")
                    with col_atm3:
                        st.markdown("**雰囲気の理由**")
                        st.success(f"{atmosphere_result.get('description', 'N/A')}")
                st.write('---')
##修正ここまで
            
            # --- 性格マッチ度分析（チャットデータがある場合のみ実行）---
            if not df.empty and 'user' in df.columns and df['user'].nunique() > 1:
                df['message'] = df['message'].fillna('')
                user_texts = df.groupby('user')['message'].apply(' '.join).reset_index()
                
                results = []
                with st.spinner('性格スコアを分析中です...'):
                    for index, row in user_texts.iterrows():
                        user = row['user']
                        text = row['message']
                        scores = analyze_personality(text)
                        dominant_personality = get_dominant_personality(scores)
                        results.append({
                            'ユーザー': user,
                            '最も強い性格傾向': dominant_personality,
                            '発言数': df[df['user'] == user].shape[0],
                            'スコア詳細': scores,
                            '全発言': text
                        })
                
                result_df = pd.DataFrame(results)

                if my_name in result_df['ユーザー'].values:
                    my_scores_row = result_df[result_df['ユーザー'] == my_name].iloc[0]
                    my_scores = my_scores_row['スコア詳細']
                    my_scores_list = list(my_scores.values())

                    match_percentages = []
                    for index, row in result_df.iterrows():
                        if row['ユーザー'] == my_name:
                            match_percentages.append(100.0)
                            continue
                        other_scores_list = list(row['スコア詳細'].values())
                        similarity = calculate_cosine_similarity(my_scores_list, other_scores_list)
                        match_percentages.append(round(similarity * 100, 2))
                    
                    result_df['自分との性格マッチ度 (%)'] = match_percentages
                    
                    ## 性格マッチ度分析結果を表示
                    st.subheader(f'👤 {my_name} とPJメンバーの性格マッチ度')
                    st.dataframe(
                        result_df[['ユーザー', '最も強い性格傾向', '発言数', '自分との性格マッチ度 (%)']]
                        .sort_values('自分との性格マッチ度 (%)', ascending=False)
                    )
                else:
                    st.warning(f"'{my_name}'のデータが分析結果に含まれていませんでした。")
                    st.dataframe(result_df[['ユーザー', '最も強い性格傾向', '発言数']])
                
                st.write('---')

                ## チャートを表示
                st.subheader('📈 全ユーザーの性格スコア比較チャート')
                if not result_df.empty:
                    fig = create_multi_user_radar_chart(result_df, my_name)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("分析対象のユーザーが見つかりませんでした。")
                
                with st.expander("各ユーザーのスコア詳細を見る"):
                    score_details_df = result_df.set_index('ユーザー')['スコア詳細'].apply(pd.Series)
                    st.dataframe(score_details_df)

                ## ペルソナ表示機能
                st.write('---')
                st.subheader('🤖 AIによるペルソナ分析')

                ## 自分以外のユーザーをマッチ度順にソート
                other_users_df = result_df[result_df['ユーザー'] != my_name].sort_values('自分との性格マッチ度 (%)', ascending=False)

                with st.spinner('AIが各メンバーのペルソナを生成中です...'):
                    # DataFrameの行をリストに変換して、インデックスでアクセスしやすくします
                    user_list = list(other_users_df.iterrows())
                    num_users = len(user_list)

                    # 2人ずつのペアで処理するためのループ (rangeの3番目の引数 `2` がステップ数)
                    for i in range(0, num_users, 2):
                        # 2つのカラムを作成
                        col1, col2 = st.columns(2)

                        # --- 1人目のペルソナを左カラムに表示 ---
                        with col1:
                            # 1人目のユーザー情報を取得
                            index1, row1 = user_list[i]
                            target_name1 = row1['ユーザー']
                            target_scores1 = row1['スコア詳細']
                            target_text1 = row1['全発言']
                            
                            # expanderを使って結果を表示 (expanded=Trueで最初から開いておく)
                            with st.expander(f"**{target_name1}さん** のペルソナを見る", expanded=True):
                                persona_dict = generate_persona_with_retry(target_name1, target_scores1, target_text1, my_scores)
                                
                                st.markdown(f"**人柄**:")
                                st.info(persona_dict.get('persona', '解析できませんでした。'))
                                
                                st.markdown(f"**自分との共通点**:")
                                st.success(persona_dict.get('common_point', '解析できませんでした。'))

                                st.markdown(f"**コミュニケーションのポイント**:")
                                st.warning(persona_dict.get('communication_point', '解析できませんでした。'))

                        # --- 2人目のペルソナを右カラムに表示 (存在する場合のみ) ---
                        if (i + 1) < num_users:
                            with col2:
                                # 2人目のユーザー情報を取得
                                index2, row2 = user_list[i + 1]
                                target_name2 = row2['ユーザー']
                                target_scores2 = row2['スコア詳細']
                                target_text2 = row2['全発言']

                                with st.expander(f"**{target_name2}さん** のペルソナを見る", expanded=True):
                                    persona_dict = generate_persona_with_retry(target_name2, target_scores2, target_text2, my_scores)
                                    
                                    st.markdown(f"**人柄**:")
                                    st.info(persona_dict.get('persona', '解析できませんでした。'))
                                    
                                    st.markdown(f"**自分との共通点**:")
                                    st.success(persona_dict.get('common_point', '解析できませんでした。'))

                                    st.markdown(f"**コミュニケーションのポイント**:")
                                    st.warning(persona_dict.get('communication_point', '解析できませんでした。'))
            else:
                 st.info("性格分析を行うには、PJメンバーのチャットファイルと自分のチャットファイルを両方アップロードしてください。")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")