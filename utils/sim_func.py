import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
import Levenshtein as LS


# Description part
CaseStudyTypeBoard= [[1.  , 0.65, 0.3 , 0.75, 0.1 , 0.1 , 0.7 , 0.1 , 0.1 , 0.2 , 0.1 , 0.7 ],
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
TaskBoard=[
            [1.  , 0.7 , 0.7 , 0.9 , 0.9 , 0.8 , 0.8 , 0.85],
            [0.7 , 1.  , 0.8 , 0.7 , 0.3 , 0.2 , 0.2 , 0.1 ],
            [0.7 , 0.8 , 1.  , 0.9 , 0.5 , 0.2 , 0.2 , 0.1 ],
            [0.9 , 0.7 , 0.9 , 1.  , 0.4 , 0.2 , 0.2 , 0.1 ],
            [0.9 , 0.3 , 0.5 , 0.4 , 1.  , 0.8 , 0.8 , 0.75],
            [0.8 , 0.2 , 0.2 , 0.2 , 0.8 , 1.  , 1.  , 0.9 ],
            [0.8 , 0.2 , 0.2 , 0.2 , 0.8 , 1.  , 1.  , 0.9 ],
            [0.85, 0.1 , 0.1 , 0.1 , 0.75, 0.9 , 0.9 , 1.  ]
            ]
CaseStudyTypeList=["Rotary machines","Structures","Production lines","Reciprocating machines","Electrical components","Lubricants",
                    "Electromechanical systems","Optical devices","Energy cells and batteries","Unknown Item Type",
                    "Pipelines and ducts","Power transmission device"]
TaskList=["Health modelling","Fault feature extraction","Fault detection","Fault identification","Health assessment",
            "One step future state forecast","Multiple steps future state forecast","Remaining useful life estimation"]  
TaxonomyTree=[ ["Fourier Transform", "Fast Fourier Transform (FFT)", "Wavelet Transform", "Spectral Mixture Kernels", 
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

def sim_leven(texto1,texto2):
    # Calcular la distancia de Levenshtein
    distancia = LS.distance(texto1, texto2)

    # Calcular la similitud de Levenshtein (normalizada)
    longitud_maxima = max(len(texto1), len(texto2))
    similitud_levenshtein = 1 - (distancia / longitud_maxima)
    return similitud_levenshtein

def sim_bin(texto1,texto2):
    return float(texto1 == texto2)
    
def sim_case_study_type(case1, case2):
    assert case1 in CaseStudyTypeList, f"Case Study type {case1} not in CaseStudyTypeList"
    assert case2 in CaseStudyTypeList, f"Case Study type  {case2} not in CaseStudyTypeList"
    i = CaseStudyTypeList.index(case1)
    j = CaseStudyTypeList.index(case2)
    return CaseStudyTypeBoard[i][i]

def sim_task(task1, task2):
    assert task1 in TaskList, f"Task1 {task1} not in TaskList"
    assert task2 in TaskList, f"Task2 {task2} not in TaskList"
    i = TaskList.index(task1)
    j = TaskList.index(task2)
    return TaskBoard[i][j]

# Solution part

def sim_taxon(model1, model2):
    Dist1=0
    Dist2=0
    Sim=0
    for i in range(0,len(TaxonomyTree)):
        for j in TaxonomyTree[i]:
            if j==model1:
                Dist1=i
            if j==model2:
                Dist2=i
    DistFin=abs(Dist2-Dist1)
    if DistFin>=6:
        Sim=0.1
    elif (DistFin<5 and DistFin>2):
        Sim=0.5
    elif DistFin<2:
        Sim=0.8
    return Sim