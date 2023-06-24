import streamlit as st

import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer


from sklearn.metrics.pairwise import cosine_similarity

def getRecommendList(text):
  credits_df = pd.read_csv("./data/credits.csv")
  movies_df = pd.read_csv("./data/movies.csv")
  
  pd.set_option("display.max_columns", None)
  pd.set_option("display.max_rows", None)
  
  # titleカラムをidとして結合する。その他のカラムは足される。
  movies_df = movies_df.merge(credits_df, on="title")
  
  movies_df = movies_df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
  
  # 欠損値の合計
  movies_df.isnull().sum()
  
  movies_df.dropna(inplace=True)

  # オブジェクトを取得する関数
  # print(movies_df.iloc[0].genres)

  # リテラル型をliteral_evalはpythonの構文として評価するために利用する。
  def convert(object):
    L = []
    for i in ast.literal_eval(object):
      L.append(i["name"])
    return L

  movies_df["genres"] = movies_df["genres"].apply(convert)
  movies_df["keywords"] = movies_df["keywords"].apply(convert)

  def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
      if counter != 3:
        L.append(i["name"])
        counter += 1
      else: 
        break
    return L

  movies_df["cast"] = movies_df["cast"].apply(convert3)

  def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
      if i["job"] == "Director":
        L.append(i["name"])
    return L

  movies_df["crew"] = movies_df["crew"].apply(fetch_director)

  movies_df["overview"] = movies_df["overview"].apply(lambda x:x.split())

  # trim
  movies_df["genres"] = movies_df["genres"].apply(lambda x: [i.replace(" ", "") for i in x] if x is not None else x)
  movies_df["keywords"] = movies_df["keywords"].apply(lambda x:[i.replace(" ", "") for i in x] if x is not None else x)
  movies_df["cast"] = movies_df["cast"].apply(lambda x:[i.replace(" ", "") for i in x] if x is not None else x)
  movies_df["crew"] = movies_df["crew"].apply(lambda x:[i.replace(" ", "") for i in x] if x is not None else x)

  movies_df["tags"] = movies_df["overview"] + movies_df["genres"] + movies_df["keywords"] + movies_df["cast"] + movies_df["crew"]

  new_df = movies_df[["movie_id", "title", "tags"]]

  new_df["tags"] = new_df["tags"].apply(lambda x:" ".join(x) if isinstance(x, list) else x)

  # 小文字にする
  new_df["tags"] = new_df["tags"].apply(lambda x:x.lower() if isinstance(x, str) else x)



  cv = CountVectorizer(max_features=5000, stop_words="english")


  ps = PorterStemmer()

  def stem(text):
    y=[]
    for i in text.split():
      y.append(ps.stem(i))
    return " ".join(y)

  # transformでnp.nanを扱えないので、から文字列で置き換える
  new_df["tags"] = new_df["tags"].fillna('')

  new_df["tags"] = new_df["tags"].apply(stem)

  vectors = cv.fit_transform(new_df["tags"]).toarray()

  similarity = cosine_similarity(vectors)

  def recommend(movie):
    movie_row = new_df[new_df["title"] == movie]
    
    if movie_row.empty:
      return None

    movie_index = movie_row.index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    L = []
    for i in movie_list:
      L.append(new_df.iloc[i[0]])
    return L
  return recommend(text)

def onClick(input_text):
  target_movie_list = getRecommendList(input_text)
  if target_movie_list is None:  # target_movie_listがNoneの場合にエラーメッセージを表示
    st.write("該当する映画が見つかりませんでした。")
  else:
    for movie in target_movie_list:
      st.write(f'映画名: {movie.title}')
      st.write(f'その他: {movie.tags}')
      st.write('=====================')


# テキストボックスの作成
input_text = st.text_input('テキストを入力してください')

# ボタンの作成
if st.button('送信'):
  onClick(input_text)
