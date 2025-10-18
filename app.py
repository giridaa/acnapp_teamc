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


# --- (新規追加) 勤怠データ分析ロジック ---
def evaluate_work_environment(df):
    """個人の勤怠データを分析し、整形済み結果と生データを返す"""
    # 1. データ準備
    df['Work start'] = pd.to_datetime(df['Work start'], errors='coerce')
    df['Work end'] = pd.to_datetime(df['Work end'], errors='coerce')
    df['Break start'] = pd.to_datetime(df['Break start'], errors='coerce')
    df['Break end'] = pd.to_datetime(df['Break end'], errors='coerce')
    df.dropna(subset=['Work start', 'Work end'], inplace=True)

    df['Duration'] = df['Work end'] - df['Work start']
    df['Break Duration'] = df['Break end'] - df['Break start']
    df['Break Duration'].fillna(pd.Timedelta(seconds=0), inplace=True)
    df['Real_Duration'] = df['Duration'] - df['Break Duration']

    daily_summary = df.groupby(df['Work start'].dt.date).agg(
        Total_Work_Duration=('Real_Duration', 'sum'),
        Total_Break_Duration=('Break Duration', 'sum')
    ).reset_index()
    daily_summary.columns = ['Date', 'Total Work Duration', 'Total Break Duration']

    # 2. 生データの計算
    daily_summary['Daily_Overtime'] = daily_summary['Total Work Duration'] - pd.Timedelta(hours=8)
    daily_summary['Daily_Overtime'] = daily_summary['Daily_Overtime'].apply(lambda x: max(x, pd.Timedelta(0)))
    
    high_daily_overtime_days = daily_summary[daily_summary['Daily_Overtime'] >= pd.Timedelta(hours=5)]
    is_dangerous_by_daily_ot = False
    if not high_daily_overtime_days.empty:
        high_daily_overtime_days = high_daily_overtime_days.copy()
        high_daily_overtime_days['Date'] = pd.to_datetime(high_daily_overtime_days['Date'])
        monthly_high_ot_counts = high_daily_overtime_days.groupby(high_daily_overtime_days['Date'].dt.to_period('M')).size()
        if (monthly_high_ot_counts >= 4).any():
            is_dangerous_by_daily_ot = True

    overtime_days_full = daily_summary[daily_summary['Total Work Duration'] > pd.Timedelta(hours=8)]
    num_overtime_days = len(overtime_days_full)

    daily_summary['Date'] = pd.to_datetime(daily_summary['Date'])
    if daily_summary.empty:
        return {}, {} 
        
    weekly_work_hours = daily_summary.groupby([daily_summary['Date'].dt.isocalendar().year, daily_summary['Date'].dt.isocalendar().week])['Total Work Duration'].sum().reset_index()
    weekly_work_hours.columns = ['Year', 'Week', 'Total Weekly Duration']
    weekly_work_hours['Overtime'] = weekly_work_hours['Total Weekly Duration'] - pd.Timedelta(hours=40)
    weekly_work_hours['Overtime'] = weekly_work_hours['Overtime'].apply(lambda x: max(x, pd.Timedelta(0)))

    def get_week_start_date(row):
        return pd.to_datetime(f'{int(row["Year"])}-W{int(row["Week"])}-1', format='%G-W%V-%u')
    weekly_work_hours['Week_Start_Date'] = weekly_work_hours.apply(get_week_start_date, axis=1)

    average_weekly_overtime = weekly_work_hours['Overtime'].mean()
    avg_overtime_hours = average_weekly_overtime.total_seconds() / 3600 if pd.notna(average_weekly_overtime) else 0

    dangerous_weeks = weekly_work_hours[weekly_work_hours['Overtime'] >= pd.Timedelta(hours=10)]
    is_dangerous_by_weekly_ot = not dangerous_weeks.empty
    is_dangerous = is_dangerous_by_weekly_ot or is_dangerous_by_daily_ot

    inadequate_break_days = overtime_days_full[overtime_days_full['Total Break Duration'] < pd.Timedelta(hours=1)]
    num_inadequate_break_days = len(inadequate_break_days)

    # 3. 傾向と理由の判定
    overtime_trend = "通常"
    trend_reason = "週平均残業時間・残業日数が基準の範囲内です"
    if is_dangerous:
        overtime_trend = "過重労働傾向あり"
        reasons = []
        if is_dangerous_by_weekly_ot:
            reasons.append("週残業10h超")
        if is_dangerous_by_daily_ot:
            reasons.append("月間5h超残業が4回以上")
        trend_reason = "、".join(reasons) + "のため"
    elif avg_overtime_hours >= 5:
        overtime_trend = "残業傾向あり"
        trend_reason = "週平均残業時間が5時間を超えているため"

    # 4. 結果の組み立て
    raw_results = {
        'avg_overtime_hours': avg_overtime_hours,
        'num_overtime_days': num_overtime_days,
        'num_inadequate_break_days': num_inadequate_break_days,
        'is_dangerous': is_dangerous
    }

    display_results = {
        "残業評価": overtime_trend,
        "評価理由": trend_reason,
        "週平均残業時間": f"{avg_overtime_hours:.2f} 時間",
        "残業日数": f"{num_overtime_days}日",
        "休憩不足の日数(8h超労働/1h未満休憩)": f"{num_inadequate_break_days}日",
        "残業が多い週(週残業10h超)": "なし",
        "長時間残業日(残業5h超/1日)": "なし"
    }
    if is_dangerous:
        if is_dangerous_by_weekly_ot:
            display_results["残業が多い週(週残業10h超)"] = ", ".join(dangerous_weeks['Week_Start_Date'].dt.strftime('%Y/%m/%d週').tolist())
        if is_dangerous_by_daily_ot:
            display_results["長時間残業日(残業5h超/1日)"] = "月に4回以上発生"

    return display_results, raw_results

def generate_chart_data(all_dfs):
    """複数人の勤怠データからプロジェクト全体の週次推移チャート用データを生成する"""
    if not all_dfs:
        return pd.DataFrame()

    all_weekly_data = []
    
    for df in all_dfs:
        # 1. 個人のDFごとにデータ準備
        df['Work start'] = pd.to_datetime(df['Work start'], errors='coerce')
        df['Work end'] = pd.to_datetime(df['Work end'], errors='coerce')
        df['Break start'] = pd.to_datetime(df['Break start'], errors='coerce')
        df['Break end'] = pd.to_datetime(df['Break end'], errors='coerce')
        df.dropna(subset=['Work start', 'Work end'], inplace=True)
        
        df['Duration'] = df['Work end'] - df['Work start']
        df['Break Duration'] = df['Break end'] - df['Break start']
        df['Break Duration'].fillna(pd.Timedelta(seconds=0), inplace=True)
        df['Real_Duration'] = df['Duration'] - df['Break Duration']

        daily_summary = df.groupby(df['Work start'].dt.date).agg(
            Total_Work_Duration=('Real_Duration', 'sum')
        ).reset_index()
        daily_summary.columns = ['Date', 'Total Work Duration']
        daily_summary['Date'] = pd.to_datetime(daily_summary['Date'])

        if daily_summary.empty:
            continue

        # 2. 個人の週次サマリー作成
        weekly_summary = daily_summary.groupby([
            daily_summary['Date'].dt.isocalendar().year,
            daily_summary['Date'].dt.isocalendar().week
        ]).agg(
            Total_Work_Duration=('Total Work Duration', 'sum'),
            OT_Days=('Total Work Duration', lambda x: (x > pd.Timedelta(hours=8)).sum())
        ).reset_index()
        weekly_summary.columns = ['Year', 'Week', 'Total Weekly Duration', 'OT_Count']
        
        weekly_summary['Overtime'] = weekly_summary['Total Weekly Duration'] - pd.Timedelta(hours=40)
        weekly_summary['Overtime'] = weekly_summary['Overtime'].apply(lambda x: max(x, pd.Timedelta(0)))
        
        all_weekly_data.append(weekly_summary)

    if not all_weekly_data:
        return pd.DataFrame()

    # 3. 全員の週次サマリーを結合して平均化
    combined_weekly = pd.concat(all_weekly_data, ignore_index=True)
    
    project_avg_weekly = combined_weekly.groupby(['Year', 'Week']).agg(
        Overtime=('Overtime', 'mean'),
        OT_Count=('OT_Count', 'mean')
    ).reset_index()

    # 4. チャート用のDataFrameを作成
    def get_week_start_date(row):
        return pd.to_datetime(f'{int(row["Year"])}-W{int(row["Week"])}-1', format='%G-W%V-%u')
    project_avg_weekly['Week_Start_Date'] = project_avg_weekly.apply(get_week_start_date, axis=1)
    
    if project_avg_weekly.empty:
        return pd.DataFrame()

    project_avg_weekly.set_index('Week_Start_Date', inplace=True)
    
    latest_date = project_avg_weekly.index.max()
    six_months_ago = latest_date - pd.DateOffset(months=6)
    
    full_date_range = pd.date_range(start=six_months_ago, end=latest_date, freq='W-MON')
    
    project_avg_weekly = project_avg_weekly.reindex(full_date_range, fill_value=0)
    
    if 'Overtime' in project_avg_weekly.columns:
         project_avg_weekly['Overtime'] = pd.to_timedelta(project_avg_weekly['Overtime'])

    project_avg_weekly = project_avg_weekly.sort_index()

    chart_df = pd.DataFrame({
        '週平均残業時間(h)': (project_avg_weekly['Overtime'].dt.total_seconds() / 3600),
        '週平均残業日数': project_avg_weekly['OT_Count']
    })
    
    return chart_df


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

# --- 4. Streamlitアプリケーションの画面 ---
st.title('📊 アサイン検討PJの分析アプリ')
st.write('アサイン予定のPJメンバーのチャットデータ、MTG会話データ、勤怠データとあなたのチャットデータ(CSV)をアップロードすると、チームの雰囲気や労働環境、メンバーの性格傾向を分析し、PJとあなたのマッチングを診断します。')
st.write('---')

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.subheader("🗫 PJチャット")
    chat_files = st.file_uploader("チャットCSVを選択", type="csv", accept_multiple_files=True, key="chat_uploader")
with col2:
    st.subheader("🗣 PJのMTG会話")
    transcript_files = st.file_uploader("音声テキストCSVを選択", type="csv", accept_multiple_files=True, key="transcript_uploader")
with col3:
    st.subheader("🗨 自分のチャット")
    my_file = st.file_uploader("自分のチャットCSVを選択", type="csv", accept_multiple_files=False, key="mychat_uploader")
with col4:
    st.subheader("🏢 PJの勤怠データ")
    work_files = st.file_uploader("勤怠データCSVを選択", type="csv", accept_multiple_files=True, key="work_uploader")
st.write('---')

if (chat_files or transcript_files or work_files) and my_file:
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

        # (新規追加) 勤怠データの読み込み
        all_member_work_dfs = []
        if work_files:
            for file in work_files:
                file.seek(0)
                try: df_single = pd.read_csv(file, encoding='shift_jis')
                except UnicodeDecodeError:
                    file.seek(0)
                    df_single = pd.read_csv(file, encoding='utf-8')
                # ユーザー名をファイル名から取得（仮）CSVに'user'列があればそちらを優先
                if 'user' not in df_single.columns:
                    df_single['user'] = os.path.splitext(file.name)[0]
                all_member_work_dfs.append(df_single)

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
        st.success(f'{len(chat_files) + len(transcript_files) + len(work_files) + 1}個のファイルの読み込みに成功！')

        if st.button('分析を実行する'):
            st.write('---'); st.header('分析結果')
            atmosphere_result, result_df = None, pd.DataFrame() # 結果を保存する変数を初期化

            # --- チームの雰囲気分析 ---
            if transcript_text:
                st.subheader('🤔 PJチームの雰囲気')
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
                    st.subheader(f'🫡 {my_name} とPJメンバーの性格マッチ度')
                    st.dataframe(result_df[['ユーザー', '最も強い性格傾向', '発言数', '自分との性格マッチ度 (%)']].sort_values('自分との性格マッチ度 (%)', ascending=False))
                    st.write('---')
                    st.subheader('📈 自分とPJメンバーの性格比較チャート')
                    st.plotly_chart(create_multi_user_radar_chart(result_df, my_name), use_container_width=True)
                    st.write('---')
                    st.subheader('🎭 PJメンバーのペルソナ分析')
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

            # --- (新規追加) 勤怠データ分析 ---
            if all_member_work_dfs:
                st.write('---')
                st.subheader('🏢 PJチームの労働環境')
                with st.spinner('勤怠データを分析中です...'):
                    individual_results = []
                    # ユーザーごとに勤怠データを結合して分析
                    all_work_df_combined = pd.concat(all_member_work_dfs, ignore_index=True)
                    unique_users = all_work_df_combined['user'].unique()
                    
                    user_dfs_for_chart = []
                    for user in unique_users:
                        user_df = all_work_df_combined[all_work_df_combined['user'] == user].copy()
                        user_dfs_for_chart.append(user_df)
                        display_res, raw_res = evaluate_work_environment(user_df)
                        individual_results.append({
                            'user': user,
                            'display': display_res,
                            'raw': raw_res
                        })

                    # プロジェクト全体の評価
                    num_individuals = len(individual_results)
                    avg_weekly_ot = sum(res['raw']['avg_overtime_hours'] for res in individual_results) / num_individuals
                    avg_overtime_days = sum(res['raw']['num_overtime_days'] for res in individual_results) / num_individuals
                    avg_inadequate_break = sum(res['raw']['num_inadequate_break_days'] for res in individual_results) / num_individuals

                    project_overtime_trend = "通常"
                    project_trend_reason = "プロジェクト全体で基準の範囲内です"
                    is_project_dangerous = any(res['raw']['is_dangerous'] for res in individual_results)
                    
                    if is_project_dangerous:
                        project_overtime_trend = "危険な労働環境"
                        project_trend_reason = "個人評価に「過重労働傾向あり」のメンバーが含まれています"
                    elif avg_weekly_ot >= 5:
                        project_overtime_trend = "残業傾向あり"
                        project_trend_reason = "プロジェクトの週平均残業時間が5時間を超えています"
                    
                    st.info(f"**プロジェクト全体の残業評価: {project_overtime_trend}** ({project_trend_reason})")
                    
                    # チャート表示
                    chart_df = generate_chart_data(user_dfs_for_chart)
                    if not chart_df.empty:
                        st.line_chart(chart_df)

                    # 個人別評価の表示
                    for res in individual_results:
                        with st.expander(f"**{res['user']}さん** の勤怠状況詳細"):
                            st.table(pd.DataFrame([res['display']]))

            # --- 総合評価 ---
            st.write('---')
            st.header('👉 総合評価')

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

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")