# AAI
## Exp:1 Implementation of Bayesian Network:
```py
!pip install pybbn

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController #Set Pandas options to display more columns
pd.options.display.max_columns=50
df=pd.read_csv('weatherAUS.csv', encoding='utf-8')
df=df[pd.isnull(df['RainTomorrow'])==False]
df=df.fillna(df.mean())
df['WindGustSpeedCat']=df['WindGustSpeed'].apply(lambda x: '0.<=40' if x<=40 else 
                                                   '1.40-50'  if 40<x<=50 else '2.>50')
df['Humidity9amCat' ]=df[ 'Humidity9am'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
df['Humidity3pmCat']=df['Humidity3pm'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
print(df)

def probs(data, child, parent1=None, parent2=None):
  if parent1==None:
    prob=pd.crosstab(data[child],'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
  elif parent1!=None:
    if parent2==None:
      prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
      prob=pd. crosstab([data[parent1], data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
  else: print("Error in Probability Frequency Calculations")
  return prob
H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child= 'Humidity3pmCat', parent1='Humidity9amCat'))
W =BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))

bbn= Bbn() \
  .add_node(H9am) \
  .add_node(H3pm) \
  .add_node(W) \
  .add_node(RT) \
  .add_edge(Edge(H9am,H3pm, EdgeType.DIRECTED)) \
  .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
  .add_edge(Edge(W,RT, EdgeType.DIRECTED))

join_tree =InferenceController.apply(bbn)
pos={0: (-1,-2), 1: (-1, 0.5), 2: (1, 0.5), 3:(0,-1)}
options ={
"font_size": 16,
"node_size": 4000,
"node_color": "white",
"edgecolors": "black",
"edge_color": "red",
"linewidths": 5,
"width": 5,
}
n,d=bbn.to_nx_graph()
nx.draw(n, with_labels=True,labels=d, pos=pos, **options)

ax=plt.gca()
ax.margins (0.20)
plt.axis("off")
plt.show()
```
## Exp:2 Implementation of Bayesian classifier:
```py
import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
class BayesClassifier:
  def __init__(self):
    self.clf = GaussianNB()
  def fit(self, X, y):
    self.clf.fit(X, y)
  def predict(self, X):
    return self.clf.predict(X)
iris=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state = 38)
clf = BayesClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 
print("Accuracy: ",accuracy)
```
## Exp:3 Implementation of Exact inference method by Bayes Network:
```py
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
network=BayesianNetwork([('Burglary','Alarm'),('Earthquake','Alarm'),
                          ('Alarm','JohnCalls'),
                         ('Alarm','MarryCalls')])
cpd_burglary=TabularCPD(variable='Burglary',variable_card=2,values=[[0.999],[0.001]])
cpd_earthquake=TabularCPD(variable='Earthquake',variable_card=2,values=[[0.998],[0.002]])
cpd_alarm=TabularCPD(variable='Alarm',variable_card=2,values=[[0.999,0.71,0.06,0.05],
            [0.001,0.29,0.94,0.95]],evidence=['Burglary','Earthquake'],evidence_card=[2,2])
cpd_john_calls=TabularCPD(variable='JohnCalls',variable_card=2,values=
                    [[0.95,0.1],[0.05,0.9]],evidence=['Alarm'],evidence_card=[2])
cpd_marry_calls=TabularCPD(variable='MarryCalls',variable_card=2,values=
                    [[0.99,0.3],[0.01,0.7]],evidence=['Alarm'],evidence_card=[2])
network.add_cpds(cpd_burglary,cpd_earthquake,cpd_alarm,cpd_john_calls,cpd_marry_calls)
inference=VariableElimination(network)
evidence={'JohnCalls':1,'MarryCalls':0}
query_variable='Burglary'
result=inference.query(variables=[query_variable],evidence=evidence)
print(result)
```
### Exp:4 Implementation of Approximate Inference in Bayesian Networks:
```py
# Import the necessary libraries

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import GibbsSampling
import networkx as nx
import matplotlib.pyplot as plt

# Define the Bayesians network Structure

network=BayesianNetwork([('Burglary','Alarm'),
                         ('Earthquake','Alarm'),
                         ('Alarm','JohnCalls'),
                         ('Alarm','MaryCalls')])

# Define the conditional Probability Distractions(CPDs)

cpd_burglay=TabularCPD(variable='Burglary',variable_card=2,values=[[0.999],[0.001]])
cpd_earthquake=TabularCPD(variable='Earthquake',variable_card=2,values=[[0.998],[0.002]])
cpd_alarm=TabularCPD(variable='Alarm',variable_card=2,values=[[0.999,0.71,0.06,0.05],
      [0.001,0.29,0.94,0.95]],evidence=['Burglary','Earthquake'],evidence_card=[2,2])
cpd_john_calls=TabularCPD(variable='JohnCalls',variable_card=2,values=
                        [[0.95,0.1],[0.05,0.9]],evidence=['Alarm'],evidence_card=[2])
cpd_mary_calls=TabularCPD(variable='MaryCalls',variable_card=2,values=
                        [[0.99,0.3],[0.01,0.7]],evidence=['Alarm'],evidence_card=[2])

# Add CPDs to the network

network.add_cpds(cpd_burglay,cpd_earthquake,cpd_alarm,cpd_john_calls,cpd_mary_calls)

# Print the Bayesian network structure

print("Bayesian Network Structure:")
print(network)

# Create a Directed Graph

G=nx.DiGraph()

# Define nodes and Edges

nodes=['Burglary', 'Earthquake', 'Alarm',' JohnCalls',' MaryCalls']
edges=[('Burglary','Alarm'),('Earthquake','Alarm'),('Alarm','JohnCalls'),('Alarm','MaryCalls')]

### Add nodes and Edges to the Graph

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Set the positions from nodes

pos={'Burglary':(0,0),'Earthquake':(2,0),'Alarm':(1,-2),'JohnCalls':(0,-4),'MaryCalls':(2,-4)}

# Draw the network

nx.draw(G,pos,with_labels=True,node_size=1500,node_color='skyblue',font_size=10,
                                                font_weight='bold',arrowsize=20)
plt.title("Bayesian Network:alarm Problem")
plt.show()

# Initialize Gibbs Sampling for MCMC

gibbs_sampler=GibbsSampling(network)

# Set the number of samples

num_samples=10000

# Perfrom MNMC sampling

samples=gibbs_sampler.sample(size=num_samples)

# Calculate approximate probabilities based on the samples

query_variable='Burglary'
query_result=samples[query_variable].value_counts(normalize=True)

# Print the approximate probabilities

print('\n Approximate probabilities of {}:'.format(query_variable))
print(query_result)
```
## Exp:5 Implementation of Hidden Markov Model:
```py
import numpy as np
transition_matrix = np.array([[0.7,0.3],
                              [0.4,0.6]])
emission_matrix = np.array([[0.1,0.9],
                            [0.8,0.2]])
initial_probablities = np.array([0.5,0.5])
observed_sequence = np.array([1,1,1,0,0,1])
alpha = np.zeros((len(observed_sequence),len(initial_probablities)))
alpha[0,:] = initial_probablities * emission_matrix[:,observed_sequence[0]]
for t in range(1,len(observed_sequence)):
  for j in range(len(initial_probablities)):
    alpha[t,j] = emission_matrix[j,observed_sequence[t]] *np.sum(alpha[t-1,:] * transition_matrix[:,j])
probablity = np.sum(alpha[-1,:])
print("The probablity of the observed sequence is:",probablity)
most_likely_sequence = []
for t in range(len(observed_sequence)):
  if alpha[t,0]>alpha[t,1]:
    most_likely_sequence.append("Sunny")
  else:
    most_likely_sequence.append("Rainy")
print("The most likely sequence of weather states is:",most_likely_sequence)
```
## Exp:6 Implementation of Kalman Filter:
```py
import numpy as np

class KalmanFilter:
  def __init__(self,F,H,Q,R,x0,P0):
    self.F=F
    self.H=H
    self.Q=Q
    self.R=R
    self.X=x0
    self.P=P0

  def predict(self):
    self.X=np.dot(self.F,self.X)
    self.P=np.dot(np.dot(self.F,self.P),self.F.T)+self.Q

  def update(self,z):
    y=z-np.dot(self.H,self.X)
    S=np.dot(np.dot(self.H,self.P),self.H.T)+self.R
    K=np.dot(np.dot(self.P,self.H.T),np.linalg.inv(S))
    self.X=self.X+np.dot(K,y)
    self.P=np.dot(np.eye(self.F.shape[0])-np.dot(K,self.H),self.P)

dt=0.1
F=np.array([[1,dt],[0,1]])
H=np.array([[1,0]])
Q=np.diag([0.1,0.1])
R=np.array([[1]])
x0=np.array([0,0])
P0=np.diag([1,1])

kf=KalmanFilter(F,H,Q,R,x0,P0)

true_states=[]
measurements=[]
for i in range(100):
  true_states.append([i*dt,1])
  measurements.append(i*dt+np.random.normal(scale=1))

est_states=[]
for z in measurements:
  kf.predict()
  kf.update(np.array([z]))
  est_states.append(kf.X)

import matplotlib.pyplot as plt
plt.plot([s[0] for s in true_states],label='true')
plt.plot([s[0] for s in est_states],label='estimate')
plt.legend
plt.show()
```
## Exp:7 Implementation of speech recognition:
```py
import speech_recognition as sr

# Assign a string variable "file" with the name of the audio file that you want to transcribe.
file = "audio.wav"

# Create an instance of the Recognizer class called "r".
r = sr.Recognizer()

# Use the AudioFile() method of sr to create an AudioFile object with the audio file name passed as an argument.
with sr.AudioFile(file) as source:
    audio = r.record(source)

# Use the recognize_google() method of r to transcribe the audio data stored in the "audio" variable.
try:
    text = r.recognize_google(audio)
except sr.UnknownValueError:
    print("Not clear")
except sr.RequestError as e:
    print("Couldn't get results from Google Speech Recognition service; {0}".format(e))

# Print the text in the next lines.
for line in text.splitlines():
    print(line)
```
## Exp:8 Implementation of sematic analysis:
```py
import nltk
from nltk.corpus import wordnet

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

# Function to identify verbs in a sentence
def get_verbs(sentence):
    verbs = []
    pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence))
    for word, tag in pos_tags:
        if tag.startswith('V'):  # Verbs start with 'V' in the POS tag
            verbs.append(word)
    return verbs


def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms


def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


def main():
    file_path = 'sample.txt'

    text = read_text_file(file_path)
    sentences = nltk.sent_tokenize(text)

    all_verbs = []
    synonyms_dict = {}

    for sentence in sentences:
        verbs = get_verbs(sentence)
        all_verbs.extend(verbs)
        for verb in verbs:
            synonyms = get_synonyms(verb)
            synonyms_dict[verb] = synonyms

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Verb', 'Synonyms'])
        for verb, synonyms in synonyms_dict.items():
            writer.writerow([verb, ', '.join(synonyms)])


if __name__ == '__main__':
    main()
```
