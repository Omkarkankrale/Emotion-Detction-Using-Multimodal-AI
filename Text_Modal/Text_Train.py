import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("D:/Emotion Detetion/MELD-RAW/MELD.Raw/test_sent_emo.csv")
df.head()

df['Emotion'].value_counts()

# sns.countplot(x='Emotion',data=df)
# plt.show()

df['Clean_Text'] = df['Utterance'].apply(nfx.remove_userhandles)

print(dir(nfx))
print("\n\n")

df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

print(df[['Utterance','Clean_Text']])

x = df['Clean_Text']
y = df['Emotion']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(x_train,y_train)
pipe_lr.score(x_test,y_test)


pipe_svm = Pipeline(steps=[('cv',CountVectorizer()),('svc', SVC(kernel = 'rbf', C = 10))])
pipe_svm.fit(x_train,y_train)
pipe_svm.score(x_test,y_test)

pipe_rf = Pipeline(steps=[('cv',CountVectorizer()),('rf', RandomForestClassifier(n_estimators=10))])
pipe_rf.fit(x_train,y_train)
pipe_rf.score(x_test,y_test)


pipeline_file = open("text_emotion.pkl","wb")
joblib.dump(pipe_lr,pipeline_file)
pipeline_file.close()