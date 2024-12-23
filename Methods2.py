import pandas as pd
import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt

import seaborn as sns # librerías para EDA visual

#Nuevo
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm
import Levenshtein
import random
#API for text semantic text similarity
#word2vec_model = api.load("glove-wiki-gigaword-50")

#Case Base comes from an Excel provided by the supervisor
path = r'C:\Users\emman\Documents\TEC\DLIG\Case Based Reasoning\CaseBase\CleanedDATA V12-05-2021.csv'
df = pd.read_csv(path, sep=';', encoding='windows-1252')

#Each case will be represented by a structure which will have the solutions, descriptions and performance seperated
class CasoInd:
    def __init__(self, Description, Solution, Performance,reference):
        #Cada uno de estos atributos son listas, menos reference que es un int
        self.description = Description
        self.solution = Solution
        self.performance=Performance
        self.reference=reference
    def __str__(self):
        return f"{self.reference}"
#Then we create the casebase asigning everything needed

CaseBase=[]
for i in range(0,len(df)):
    Description=df.loc[i,['Task', 'Case study type', 'Case study', 'Online/Off-line', 'Input for the model']]
    Solution=df.loc[i,['Model Approach', 'Model Type', 'Models', 'Data Pre-processing', 'Complementary notes', 'Publication identifier,,,,,,,,,,,,,,,,,,']] #Gotta fix this bug
    Performance=df.loc[i,['Performance indicator', 'Performance', 'Publication Year']]
    case=CasoInd(Description.values,Solution.values,Performance.values,i)
    CaseBase.append(case)

class Similarity:
    def __init__(self):
        #Se utiliza esto para definir nuestra similitud con CASOS DE ESTUDIO y TAREAS
        #Porque esas son más subjetivas
        self.CaseStudyTypeBoard= [[1.  , 0.65, 0.3 , 0.75, 0.1 , 0.1 , 0.7 , 0.1 , 0.1 , 0.2 , 0.1 , 0.7 ],
                                  [0.65, 1.  , 0.3 , 0.75, 0.1 , 0.1 , 0.1 , 0.1 , 0.  , 0.2 , 0.75, 0.1 ],
                                  [0.3 , 0.3 , 1.  , 0.3 , 0.3 , 0.2 , 0.9 , 0.1 , 0.1 , 0.2 , 0.7 , 0.1 ],
                                  [0.75, 0.75, 0.3 , 1.  , 0.1 , 0.1 , 0.7 , 0.1 , 0.1 , 0.2 , 0.1 , 0.7 ],
                                  [0.1 , 0.1 , 0.3 , 0.1 , 1.  , 0.1 , 0.3 , 0.6 , 0.7 , 0.2 , 0.1 , 0.3 ],
                                  [0.1 , 0.1 , 0.2 , 0.1 , 0.1 , 1.  , 0.1 , 0.1 , 0.1 , 0.2 , 0.1 , 0.1 ],
                                  [0.7 , 0.1 , 0.9 , 0.7 , 0.3 , 0.1 , 1.  , 0.3 , 0.3 , 0.2 , 0.1 , 0.7 ],
                                  [0.1 , 0.1 , 0.1 , 0.1 , 0.6 , 0.1 , 0.3 , 1.  , 0.1 , 0.2 , 0.1 , 0.1 ],
                                  [0.1 , 0.  , 0.1 , 0.1 , 0.7 , 0.1 , 0.3 , 0.1 , 1.  , 0.2 , 0.  , 0.7 ],
                                  [0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 1.  , 0.2 , 0.2 ],
                                  [0.1 , 0.75, 0.7 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.  , 0.2 , 1.  , 0.1 ],
                                  [0.7 , 0.1 , 0.1 , 0.7 , 0.3 , 0.1 , 0.7 , 0.1 , 0.7 , 0.2 , 0.1 , 1.  ]]
        #Vamos a hacer nuevas tablas
        self.TaskBoard=[
                        [1.  , 0.7 , 0.7 , 0.9 , 0.9 , 0.8 , 0.8 , 0.85],
                        [0.7 , 1.  , 0.8 , 0.7 , 0.3 , 0.2 , 0.2 , 0.1 ],
                        [0.7 , 0.8 , 1.  , 0.9 , 0.5 , 0.2 , 0.2 , 0.1 ],
                        [0.9 , 0.7 , 0.9 , 1.  , 0.4 , 0.2 , 0.2 , 0.1 ],
                        [0.9 , 0.3 , 0.5 , 0.4 , 1.  , 0.8 , 0.8 , 0.75],
                        [0.8 , 0.2 , 0.2 , 0.2 , 0.8 , 1.  , 1.  , 0.9 ],
                        [0.8 , 0.2 , 0.2 , 0.2 , 0.8 , 1.  , 1.  , 0.9 ],
                        [0.85, 0.1 , 0.1 , 0.1 , 0.75, 0.9 , 0.9 , 1.  ]
                        ]
        
        self.CaseStudyTypeList=["Rotary machines","Structures","Production lines","Reciprocating machines","Electrical components","Lubricants",
                                "Electromechanical systems","Optical devices","Energy cells and batteries","Unknown Item Type",
                                "Pipelines and ducts","Power transmission device"]
        
        self.TaskList=["Health modelling","Fault feature extraction","Fault detection","Fault identification","Health assessment",
                       "One step future state forecast","Multiple steps future state forecast","Remaining useful life estimation"]
        
        self.TaxonomyTree=[ ["Fourier Transform", "Fast Fourier Transform (FFT)", "Wavelet Transform", "Spectral Mixture Kernels", 
                            "Multiscale Gaussian Process Regression (SE-MGPR)", "Gaussian Process Functional Regression with Periodic Covariance",
                              "Time Continuous Relevant Isometric Mapping (TRIM)", "Principal Component Analysis (PCA)", 
                              "Dynamic Time Warping (DTW)", "Capacitance Degradation Dynamic Model", "Electrochemical-Thermal Model", "Spall Initiation Model", "Spall Progression Model"],
                            ["Physics-based model for track settlement (Bayes)", "Physics-based model for track settlement (Bayes, Particle Filter)",
                                "Paris Law", "Wiener Process", "Modified Wiener Process", "Ornstein-Uhlenbeck Process", "Gamma Process", "Compound Poisson",
                                "Filter Clogging Model", "Physical Erosion Model", "Damage Accumulation Model", "Structural Reliability Analysis", 
                                "Physics of Failure Descriptive Model"],
                            ["Hybrid Kalman Filter with OBEM Model", "Particle Filter with ANFIS", "Markov Chain with Bayesian Networks", 
                             "Support Vector Regression with Particle Filter", "Hidden Markov Model with Genetic Algorithm", 
                             "Markov Chain with Neural Networks", "Relevance Vector Machine with Kalman Filter", 
                             "Rule-Based Systems with Kalman Filter", "Gaussian Process Functional Regression (GPFR) with Neural Networks", 
                             "Gauss-Markov Processes with Dynamical Models for Rotating Machines"],
                            ["Monte Carlo Simulation", "Sequential Monte Carlo (SMC)", "Particle Filter with Monte Carlo Method", 
                             "Inverse FORM (First Order Reliability Method)", "Genetic Algorithm (GA)", "Copula-Based Sampling", 
                             "State Projection Scheme with Particle Filter", "Trajectory Similarity Optimization", "Spall Initiation and Progression Models", 
                             "Likelihood Probability Weight Calculation"],
                            ["Bagging", "Boosting", "Random Forest", "Ensemble Learning-based Prognostic Model", "Voting Method", 
                             "AEKFOS-ELM (Adaptive Weighted Ensemble of KFOS-ELM)", "EOS-ELM (Ensemble of OS-ELM)", 
                             "Physics-based model with Bayesian and Particle Filter Ensembles", "Relevance Vector Machine and Kalman Filter Ensembles", 
                             "Ensemble Learning-based Approaches", "Extreme Gradient Boosting (XGBoost)", 
                             "Hybrid Kalman Filter with Piecewise Linear Model (PWL)", "Similarity-based Interpolation with Ensemble Learning", 
                             "Deep Belief Network with PCA (Principal Component Analysis)"],
                            ["Fuzzy Inference System", "Rule-Based System", "Rule-Based Model with Kalman Filter", "Fuzzy Neural Network",
                              "Fuzzy Inference System with Neural Network", "Fuzzy Inference System with SVM Regression", "Belief Rule-Based Model", 
                              "Updated Rule-Based System", "Piecewise Model", "Point-Wise Fuzzy Model"],
                            ["Logistic Regression", "Bayesian Linear Regression", "Least Squares Exponential Fitting", "Statistical Regression Model", 
                             "Support Vector Regression", "Least Square Support Vector Machine", "Bayesian Linear Regression", "Ridge Regression", 
                             "Polynomial Function", "Quadratic Function", "Recursive Least Squares", "Wiener Process", "Proportional Hazards Model", 
                             "Nonlinear Regression", "Linear Regression", "ARMA (Auto-Regressive Moving Average)", "Bayesian Linear Regression", 
                             "Rule-based system with Linear Regression", "Similarity-Based Interpolation", "Least Square Exponential Fitting"],
                            ["MLP (Multi-layer Perceptron)", "Radial Basis Function Network", "LSTM (Long Short-Term Memory)", "GRU (Gated Recurrent Unit)", 
                             "Deep Learning Network", "Auto-encoder with Deep Transfer Learning", "Deep Belief Network", "Recurrent Neural Network (RNN)", 
                             "Convolutional Neural Network (CNN)", "Sparse Filter", "MONESN (Monotonic Echo State Network)", "Back Propagation Neural Network", 
                             "Elman Neural Network", "ANFIS (Adaptive Neuro-Fuzzy Inference System)", "Deep Belief Network-based Hierarchical Diagnosis Network"],
                            ["Principal Component Analysis (PCA)", "Sparse Filter", "Proper Orthogonal Decomposition (POD)", "Supervised Machine Learning with Dimension Reduction", 
                             "Trace Ratio Linear Discriminant Analysis", "Affinity Matrix", "Inherent Feature Evaluation", "Geolocation Principal", 
                             "Recursive Least Squares with Dimensionality Reduction", "Linear Gaussian Process Functional Regression (LGPFR)", 
                             "Deep Belief Network with PCA"],
                            ["Kalman Filter", "Unscented Kalman Filter (UKF)", "Hidden Markov Models (HMM)", "Bayesian Networks", 
                             "Mixture of Gaussians Hidden Markov Models (MoG-HMM)", "Particle Filter", "Bayesian Model", "Inverse Gaussian Distribution", 
                             "Levy Distribution", "Gauss-Markov Process", "Expectation Maximization (EM)", "Monte Carlo Method", "Sequential Monte Carlo", 
                             "Gaussian Process Regression (GPR)", "Gaussian Process Functional Regression", "Hidden Markov with Genetic Algorithm", "Dempster-Shafer Theory", 
                             "Probability Approach", "Gaussian Process Regression with Neural Networks (GPRNN)", "Recursive Maximum Likelihood Estimation (RMLE)"]]
        
    def CosineSimilarity(self,text1,text2):
        documents = [text1,text2]
        count_vectorizer = CountVectorizer(stop_words="english")
        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform(documents)
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(
            doc_term_matrix,
            columns=count_vectorizer.get_feature_names_out(),
            )
        return cosine_similarity(df, df)
    # Función para obtener el embedding promedio de una oración
    def sentence_embedding(self,texto, model):
        if len(texto) == 1 and texto[0] in model:
            return model[texto[0]]  # Retorna directamente el vector de la palabra
        else:
            vector = [model[word] for word in texto if word in model]
            return sum(vector) / len(vector) if len(vector) > 0 else [0] * 50
    
    def SimSemantic(self,texto1,texto2,model):
        texto1 = texto1.split()
        texto2 = texto2.split()

        # Obtener los embeddings promedio de los dos textos
        embedding1 = self.sentence_embedding(texto1, model)
        embedding2 = self.sentence_embedding(texto2, model)

        # Calcular la similitud coseno entre los embeddings
        similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
        return similarity
    
    def Levenshtein(self,texto1,texto2):
        # Calcular la distancia de Levenshtein
        distancia = Levenshtein.distance(texto1, texto2)

        # Calcular la similitud de Levenshtein (normalizada)
        longitud_maxima = max(len(texto1), len(texto2))
        similitud_levenshtein = 1 - (distancia / longitud_maxima)
        return similitud_levenshtein


    def On_Offline(self,texto1,texto2):
        if texto1==texto2:
            Sim=1
        else:
            Sim=0
        return Sim
        
    def SimCaseStudyType(self,Case1,Case2):
        CaseStudy1=Case1.description[1]
        CaseStudy2=Case2.description[1]
        a=0
        b=0
        for i in range(0,len(self.CaseStudyTypeList)):
            if self.CaseStudyTypeList[i]==CaseStudy1:
                a=i
            elif self.CaseStudyTypeList[i]==CaseStudy2:
                b=i
        return self.CaseStudyTypeBoard[a][b]
    
    def SimTask(self,Case1,Case2):
        CaseTask1=Case1.description[0]
        CaseTask2=Case2.description[0]
        a=0
        b=0
        for i in range(0,len(self.TaskList)):
            if self.TaskList[i]==CaseTask1:
                a=i
            elif self.TaskList[i]==CaseTask2:
                b=i
        return self.TaskBoard[a][b]
    #A partir de acá es la comparación para soluciones
    def SimPrePro(self,Case1,Case2):
        if Case1.solution[4]==Case2.solution[4]:
            SimPre=1
        else:
            SimPre=0
        return SimPre
    def SimTaxon(self,Case1,Case2):
        Model1=Case1.solution[2]
        Model2=Case2.solution[2]
        Dist1=0
        Dist2=0
        Sim=0
        for i in range(0,len(self.TaxonomyTree)):
            for j in self.TaxonomyTree[i]:
                if j==Model1:
                    Dist1=i
                if j==Model2:
                    Dist2=i
        DistFin=abs(Dist2-Dist1)
        if DistFin>=6:
            Sim=0.1
        elif (DistFin<5 and DistFin>2):
            Sim=0.5
        elif DistFin<2:
            Sim=0.8
        return Sim
    
    def GlobalSim(self,Similarities:list,Weights:list):
        #SimGlob=math.sqrt(reduce(lambda x, y: x + y, [((num * peso)**2)for num, peso in zip(Similarities, Weights)]))
            # Verificamos que el número de similitudes coincida con el número de pesos
    
        # Calcular la similitud global ponderada
        similitud_global = sum(s * p for s, p in zip(Similarities, Weights))
    
        #Asegurarse de que la salida esté en el rango de 0 a 1
        similitud_global = max(0, min(similitud_global, 1))
    
        return similitud_global

def CompareSimilarityDesc(Sol1,Sol2,Weights):
    #Gives the similarity for two given cases
    Sim=Similarity()
    Sim1=Sim.SimTask(Sol1,Sol2)
    Sim2=Sim.SimCaseStudyType(Sol1,Sol2)
    Sim3=Sim.Levenshtein(Sol1.description[2],Sol2.description[2])#Case Study
    Sim4=Sim.On_Offline(Sol1.description[3],Sol2.description[3])
    Sim5=Sim.Levenshtein(Sol1.description[4],Sol2.description[4]) #Input for the model
    SimGlobal=Sim.GlobalSim([Sim1,Sim2,Sim3,Sim4,Sim5],Weights)    
    return SimGlobal

def CompareSimilaritySol(Sol1,Sol2,Weights):
    Sim=Similarity()
    Sim1=Sim.SimPrePro(Sol1,Sol2)#Data PreProcessing
    Sim2=Sim.Levenshtein(Sol1.solution[0],Sol2.solution[0])#Model Approach
    Sim3=Sim.Levenshtein(Sol1.solution[2],Sol2.solution[2])#Model Type
    Sim4=Sim.SimTaxon(Sol1,Sol2)#Model
    SimGlobal=Sim.GlobalSim([Sim1,Sim2,Sim3,Sim4],Weights)#Global Sim
    return SimGlobal


def SearchSimilar(UserInput,CaseBase,NumberRetrievals,Weights):
    InputCase=CasoInd(UserInput,0,0,"CasoUsuario")
    ListRetrievals=[]
    ListSim=[]
    for i in range(0,len(CaseBase)-1):
        Sim=CompareSimilarityDesc(InputCase,CaseBase[i],Weights)
        if i==0:
            ListRetrievals.append(CaseBase[i])
        else:
            if len(ListRetrievals)<NumberRetrievals:
                    ListRetrievals.append(CaseBase[i])
                    ListSim.append(Sim)
            else:
                 for j in range(0,NumberRetrievals-1): 
                      if Sim>ListSim[j]:
                        ListRetrievals.pop(j)
                        ListSim.pop(j)
                        ListRetrievals.append(CaseBase[i])
                        ListSim.append(Sim) 
                        break
    return ListRetrievals,ListSim

class Solution:
    def __init__(self,solution,reference):
        self.solution=solution
        self.NestedSolutions=[]
        self.link=reference

class Description:
    def __init__(self,description,reference):
        self.description=description
        self.NestedDescriptions=[]
        self.link=reference

def quicksort_objetos(Lista):
    if len(Lista) <= 1:
        return Lista
    else:
        pivote = Lista[0]  # Elegimos el primer objeto como pivote
        menores = [obj for obj in Lista[1:] if obj.link <= pivote.link]  # Objetos con link menor o igual al pivote
        mayores = [obj for obj in Lista[1:] if obj.link > pivote.link]   # Objetos con link mayor al pivote
        return quicksort_objetos(menores) + [pivote] + quicksort_objetos(mayores)
    
def DescriptionsAndSolutions(CaseBase):
    SolutionList=[]
    DescriptionList=[]
    for i in CaseBase:
        Sol=Solution(i.solution,i.reference)
        Desc=Description(i.description,i.reference)
        SolutionList.append(Sol)
        DescriptionList.append(Desc)
    return SolutionList,DescriptionList

def ModifiedCNN(SolutionList,DescriptionList,WeightsSol,WeightsDesc):

    NestedSolutionList=[]
    NestedDescriptionList=[]
    ThresholdSol=0.7 
    ThresholdDesc=0.7
    #########################################Generalization of descriptions####################################################
    i=0
    while DescriptionList:
        if i==0:
            RandDescription = random.sample(DescriptionList, 1)
            NestedDescriptionList.append(RandDescription[0])
            DescriptionList=[obj for obj in DescriptionList if obj not in RandDescription]
            i+=1
        else:
            RandDescription = random.sample(DescriptionList, 1)
            DescriptionList=[obj for obj in DescriptionList if obj not in RandDescription]
            for j in range(0,len(NestedDescriptionList)):#Compare Similarity with the rest of the descriptions
                DescriptionSimilarity=CompareSimilarityDesc(NestedDescriptionList[j],RandDescription[0],WeightsDesc)
                
                if DescriptionSimilarity>=ThresholdDesc:#if it is similar, nest it
                    NestedDescriptionList[j].NestedDescriptions.append(RandDescription[0])
                    SolutionList[RandDescription[0].link].link=NestedDescriptionList[j].link #Se asume que SolutionList está ordenada
                    #Now the solution that this one had has to be reindexed
                    break
                if j==(len(NestedDescriptionList)-1):
                    if DescriptionSimilarity<ThresholdDesc:#if it is not similar to anybody it is a GC
                        NestedDescriptionList.append(RandDescription[0])
    

    ##########################################Generalization of solutions###################################################
    i=0
    while SolutionList:
        if i==0:
            RandSolution = random.sample(SolutionList, 1)
            NestedSolutionList.append(RandSolution[0])
            SolutionList=[obj for obj in SolutionList if obj not in RandSolution]
            i+=1
        else:
            RandSolution = random.sample(SolutionList, 1)
            SolutionList=[obj for obj in SolutionList if obj not in RandSolution]
            for j in range(0,len(NestedSolutionList)):#Compare Similarity with the rest of the solutions
                SolutionSimilarity=CompareSimilaritySol(NestedSolutionList[j],RandSolution[0],WeightsSol)
                if SolutionSimilarity>=ThresholdSol:#if it is similar, nest it
                    NestedSolutionList[j].NestedSolutions.append(RandSolution[0])
                    #Now the description that this one had has to be reindexed
                    break
                if j==(len(NestedSolutionList)-1):
                    if SolutionSimilarity<ThresholdSol:#if it is not similar to anybody it is a GC
                        NestedSolutionList.append(RandSolution[0])

    #################################Process of reindexation and re-arraging of solutions###################################
    #First re-arrange the solutions
    """for k in range(0,len(NestedSolutionList)):
        if len(NestedSolutionList[k].NestedSolution)>=1:"""
            #The solution returned by checking the performance
            #Will be the GC and nest the other solutions to it
            #Check the descriptions that have
    return NestedDescriptionList,NestedSolutionList

def Performance(NestedSolutionList,GenSolution):
    #Get performance indicator from each one of the nested solutions
    #Pick the best
    #return the best solution
    #This has to be done with a metric and a case base of performance indicators
    #Because of time I will skip this, leaving this as room for improvement
    pass


def Diversity(RetrievedSolutions,weights):
    #Ill do it taking the average similarity between retrieved solutions
    listDiv=[]
    for i in range(0,len(RetrievedSolutions)-1):
        for j in range(i+1,len(RetrievedSolutions)):
            Div=1-CompareSimilaritySol(RetrievedSolutions[i],RetrievedSolutions[j],weights)
            listDiv.append(Div)
    return sum(listDiv) / len(listDiv)

def SearchSimilarModCNN(UserInput,Descrip,Solut,NumberRetrievals,Weights):
    #This function searches similar cases to the user input by searching for closest neighbors in the case base
    InputCase=Description(UserInput,299)
    ListRetrievalsDesc=[]
    ListSim=[]
    #First we compare the user input with the descriptions
    for i in range(0,len(Descrip)-1):
        Sim=CompareSimilarityDesc(InputCase,Descrip[i],Weights)
        if i==0:
            ListRetrievalsDesc.append(Descrip[i])
            ListSim.append(Sim)
        else:
            if len(ListRetrievalsDesc)<NumberRetrievals:
                    ListRetrievalsDesc.append(Descrip[i])
                    ListSim.append(Sim)
            else:
                 for j in range(0,NumberRetrievals-1): 
                      if Sim>ListSim[j]:
                        ListRetrievalsDesc.pop(j)
                        ListSim.pop(j)
                        ListRetrievalsDesc.append(Descrip[i])
                        ListSim.append(Sim) 
                        break
       
    OrderIndexList= sorted(range(len(ListSim)), key=lambda i: ListSim[i], reverse=True)
    ListRetrievalSol=[]
    for j in OrderIndexList:
        LinkSol=ListRetrievalsDesc[j].link
        for k in Solut:
            if k.link==LinkSol:
                ListRetrievalSol.append(k)
        if len(ListRetrievalSol)>NumberRetrievals:
            ListRetrievalSol=[ListRetrievalSol[0],ListRetrievalSol[1],ListRetrievalSol[2],ListRetrievalSol[3],ListRetrievalSol[4]]
            break
    return ListRetrievalSol

