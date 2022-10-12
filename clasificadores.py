#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import math
import random

#import data from archivo

from collections import Counter
import warnings

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)



#partición aleatoria y estratificada del conjunto de datos

def particion_estratificada(X, y, test = 0.2):
    
    #cuenta el número de elementos por cada clase
    c = Counter(y)
    
    #unimos cada dato con su clase correspondiente y ordenamos según la clase
    #con el objetivo de conocer el índice en la lista por el que empieza una clase
    aux = list(zip(X, y))
    aux.sort(key=lambda a: a[1])

    #elem = list()
    
    testset = list()
    trainset = list()
    
    total = sum(c.values())
    
    for e in c.keys(): #itera según la cantidad de clases
        elemsClase = c.get(aux[0][1])  #número de elementos de la primera clase de la lista
        o = aux[:elemsClase]   #obtiene de la lista todos los elemenos que tengan la primera clase
        random.shuffle(o)   #y los mezcla para añadir aleatoriedad
        
        fl = math.floor((elemsClase/total)*(test*total)) #n elementos de clase var correspondientes a test set
        
        testset.extend(o[:fl])  #los n elementos de clase var correspondientes a test set van al testset
        trainset.extend(o[fl:]) #el resto irán al trainset
        
        
        del aux[:elemsClase] #elimina todos los elementos de la clase ya procesada
    
    #aleatoriezamos las tuplas para que no queden ordenadas por clase
    np.random.shuffle(testset)
    np.random.shuffle(trainset)
    

    Xentren, yentren = zip(*trainset)
    Xprueba, yprueba = zip(*testset)
    
    
    return np.array(Xentren), np.array(Xprueba), np.array(yentren), np.array(yprueba)

        
'''
a = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,0]])
b = np.array(['si','no','si','no','no','no'])
'''

#Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(a,b,test=2/6)
#Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(data.X_votos,data.y_votos,test=1/3)
#Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(data.X_credito,data.y_credito,test=0.4)

#normaliza los datos
def normalizar(X):         
            #dat = X[0:, :]         
           
            #calcula la media y la desviación estándar para cada columna o característica (axis = 0),
            X2 = np.array(X)
            media = X2.mean(axis=0)
            std = X2.std(axis=0)

            '''
            Numpy añade columnas iguales para igualar el tamaño de ambos arrays bidimensionales
            X2              media           X2              media 
            [[0  1  2],  -  [1  2  1]  =   [[0  1  2],  -  [[1  2  1],
             [3  4  5]]                     [3  4  5]]      [1  2  1]]
            '''
            return (X2 - media) / std

class Perceptron():

    
    def __init__(self,normalizacion=False,
                 rate=0.1,rate_decay=False,n_epochs=100,
                 pesos_iniciales=None):

        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.pesos_iniciales = pesos_iniciales

    trad = dict()        
    entrenado = False
    pesos = []
    clases = []
    rate_inicial = 0
    def entrena(self,X,y):

        #almacenamos las clases
        self.clases = list(set(sorted(y)))
        
        #en el caso de que las clases no sean ya 0 y 1, las almacenamos en un diccionario
        #y las cambiamos a 0 y 1 para trabajar más fácilmente con ellas
        if self.clases != [0,1]:
            y = np.where(y == self.clases[0], 0, 1)
            self.trad[0] = self.clases[0]
            self.trad[1] = self.clases[1]
        
        #si no me dan un array de pesos iniciales...
        #genero uno con decimales aleatorios entre -1 y 1
        if self.pesos_iniciales is None:
            self.pesos_iniciales = np.random.uniform(-1, 1, len(X[0])+1) #len+1 para el peso w0
            
        self.pesos = self.pesos_iniciales
        self.rate_inicial = self.rate
        #thr = 0
        
        if self.normalizacion:
            X = normalizar(X)
            
        for i in range(self.n_epochs):
             
            #en cada iteración, tomar un ejemplo Xi
            for j in range(len(X)):
                
                #calcular clasificación para (w*Xi)
                #básicamente cambian los pesos cuando el valor predicho no es igual que el real
                o = np.dot(X[j],self.pesos[1:]) #pesos[1:] para no multiplicar por peso w0
                #o = 1 if o >= thr else 0
                o = 1 if o >= 0 else 0
                #calcula la diferencia
                dif = y[j] - o
                
                #np.insert(X[j], 0, 1) devuelve X[0] con un 1 en la posición 0, para que
                #cuadren los temaños de las listas porque pesos tiene w0
                self.pesos += self.rate * (np.insert(X[j], 0, 1) * dif)
                    
            if self.rate_decay:
                self.rate = self.rate_inicial*(1/(1+i)) 
                
        self.entrenado = True
        
        
              
    def clasifica(self,ejemplos):
        
        try:        
            if self.normalizacion:
                ejemplos = normalizar(ejemplos)
                
            s = np.dot(ejemplos, self.pesos[1:])
            s = np.where(s>=0, 1, 0)

            #solamente en caso de que ya hayamos entrenado el algoritmo, por legibilidad
            #devolvemos las clases traducidas a sus nombres originales
            if self.clases != [0,1] and self.entrenado:
                s = np.where(s == 0 , self.trad[0], self.trad[1])

            return s
        
        except:
            raise ClasificadorNoEntrenado()
            


Xe_cancer,Xp_cancer,ye_cancer,yp_cancer=particion_estratificada(data.X_cancer,data.y_cancer)
Xe_votos,Xp_votos,ye_votos,yp_votos=particion_estratificada(data.X_votos,data.y_votos,test=1/3)

'''
Xe_p, Xp_p, ye_p, yp_p = particion_estratificada(
    np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]),
    np.array([1,0,1,0,1,0,1,0]),
    test = 0.25)
'''

perc_cancer=Perceptron(rate=0.1,rate_decay=True,normalizacion=True)
perc_votos=Perceptron(rate=0.1,rate_decay=True,normalizacion=True)


perc_cancer.entrena(Xe_cancer,ye_cancer)
perc_votos.entrena(Xe_votos, ye_votos)

      

def rendimiento(perc, Xe, ye):
    clasif = perc.clasifica(Xe)
    #Devuelve correctamenteclasificados/entradastotales
    return sum(clasif==ye) / ye.shape[0]




from scipy.special import expit    
#
def sigmoide(x):
    return expit(x)


class RegresionLogisticaMiniBatch():

    def __init__(self,normalizacion=False,
                 rate=0.1,rate_decay=False,n_epochs=100,
                 pesos_iniciales=None,batch_tam=64):
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.pesos_iniciales = pesos_iniciales
        self.batch_tam = batch_tam
        
    trad = dict()        
    entrenado = False
    pesos = []
    clases = []
    rate_inicial = 0
        
    def entrena(self,X,y,salida_epoch=False):       

        #almacenamos las clases
        self.clases = list(set(sorted(y)))
        
        
        #en el caso de que las clases no sean ya 0 y 1, las almacenamos en un diccionario
        #y las cambiamos a 0 y 1 para trabajar más fácilmente con ellas
        if self.clases != [0,1]:
            print("a")
            y = np.where(y == self.clases[0], 0, 1)
            self.trad[0] = self.clases[0]
            self.trad[1] = self.clases[1]
        
        #si no me dan un array de pesos iniciales...
        #genero uno con decimales aleatorios entre -1 y 1
        if self.pesos_iniciales is None:
            self.pesos_iniciales = np.random.uniform(-1, 1, len(X[0])+1)
        
        self.pesos = self.pesos_iniciales
        self.rate_inicial = self.rate
   
        if self.normalizacion:
            X = normalizar(X)
            
        num_batches = math.floor(len(X)/self.batch_tam) #calcula el número de batches
        
        for i in range(self.n_epochs):
            
            ar = np.arange(len(X)) #lista con índices del os elementos de X
            np.random.shuffle(ar)  #mezcla aleatoriamente los índices
             
            minibatches = np.array_split(ar, num_batches)   #separa los índices en diferentes batches

            #en cada iteración, operar con minibatches
            for mb in minibatches:
                
                #índices para el minibatch actual
                X_mbatch = np.array([X[b] for b in mb])
                y_mbatch = np.array([y[b] for b in mb])

                #calcula la hipótesis probabilística de Xmbatch pertenezca a la clase 1
                h = sigmoide(np.dot(self.pesos[1:],X_mbatch.T))

                #calcula la entropía cruzada para Xmbatch 
                ent = sum(np.where(y_mbatch == 1, np.negative(np.log(h)), np.negative(np.log(1-h))))
                
                #calcula el gradiente para los ejemplos del minibatch
                grad = np.dot(X_mbatch.T, y_mbatch-h) / y_mbatch.size
                
                #actualizamos los pesos
                self.pesos = self.pesos + (self.rate * np.insert(grad, 0, 1))

            if self.rate_decay:
                self.rate = self.rate_inicial*(1/(1+i)) 
                
            if(salida_epoch):
                print("Epoch", i, ", entropía cruzada: ",ent,", rendimiento: ",rendimiento(self, X,y))
        
        
        
    #función sigmoide
    def clasifica_prob(self,ejemplos):
        try:
            if self.normalizacion:
                ejemplos = normalizar(ejemplos)
                
            return sigmoide(np.dot(self.pesos[1:],ejemplos.T))
        
        except:
            raise ClasificadorNoEntrenado()

    
    def clasifica(self,ejemplos):
        try:
            
            if self.normalizacion:
                ejemplos = normalizar(ejemplos)

            s = sigmoide(np.dot(self.pesos[1:],ejemplos.T))
            return np.where(s >= 0.5, 1, 0)
        
        except:
            raise ClasificadorNoEntrenado()



lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True,n_epochs=10)
lr_cancer.entrena(Xe_cancer,ye_cancer,salida_epoch=True)




def rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5):
    #cuenta el número de elementos por cada clase
    eXc = Counter(y)
    
    #unimos cada dato con su clase correspondiente y ordenamos según la clase
    #con el objetivo de conocer el índice en la lista por el que empieza una clase
    aux = list(zip(X, y))
    aux.sort(key=lambda a: a[1])
    
    elementos = []
    clases = []
    
    for e in eXc.keys(): #itera según la cantidad de clases
        elemsClase = eXc.get(aux[0][1])  #número de elementos de la primera clase de la lista
        o = aux[:elemsClase]   #obtiene de la lista todos los elemenos que tengan la primera clase
        random.shuffle(o)   #y los mezcla para añadir aleatoriedad
        
        e, c = zip(*o)

        elementos.append(np.array_split(e, n))
        clases.append(np.array_split(c, n))     #separa cada clase en n trozos de igual tamaño

        del aux[:elemsClase] #elimina todos los elementos de la clase ya procesada
        
    elementos = np.array(elementos)
    clases = np.array(clases)
    
    #print(elementos)
    #print(clases)
    
    
    k_folds_X = np.column_stack(elementos) #de forma similar a una transpuesta,
    k_folds_y = np.column_stack(clases)    #cambia la forma de agrupar los elementos
    #ahora cada n-trozo contiene elementos de cada clase de forma estratificada
    
    print(k_folds_X)
    print(k_folds_y)
    
    
    total = 0
    
    #itera según el número de particiones
    for k in range(n):
        
        #concatena todas las particiones menos la que vamos a utilizar como test
        Xaux = np.concatenate([k_folds_X[:k], k_folds_X[k+1:]])
        yaux = np.concatenate([k_folds_y[:k], k_folds_y[k+1:]])
        
        #quita dimensiones a los np.arrays para que no queden: [ [kfold1] , [kfold2] ]
        #sino que sea una lista de elementos [ elem1, elem2 ]
        Xaux = Xaux.reshape(-1, Xaux.shape[-1])
        yaux = yaux.flatten()
        
        clasificador = clase_clasificador(**params)
        clasificador.entrena(Xaux, yaux)
        
        total += rendimiento(clasificador, k_folds_X[k], k_folds_y[k])
    
    return total/n


class clasificador_multiclase():
    def __init__(self,normalizacion=False,
                 rate=0.1,rate_decay=False,n_epochs=100,batch_tam=64):
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.listaRL = []  
        self.clases = []
        
    def entrena(self, X, y):
        self.clases = np.unique(y);
        
        #Un clasificador de regresión logística entrenado por cada clase
        for c in self.clases:
            aux = np.where(y == c, 1, 0)
            
            rl=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True,n_epochs=10)
            rl.entrena(X, aux)
            self.listaRL.append(rl)
            
    def clasifica(self, ejemplos):
        try:
            #Clasifica en función de la clase que sea más probable según el clasificiador
            #de regresión logística que le corresponde
            probs = []
            
            #Almacena en una matriz las probabilidades de cada ejemplo perteneciendo a cada clase
            #Índice de fila: clase
            #Índice de columna: ejemplo
            for i in range(len(self.listaRL)):
                probs.append(self.listaRL[i].clasifica_prob(ejemplos))
            
            #Argmax en el eje cero me devuelve los índices (clases) de las columnas con mayor valor
            #Es decir, que se queda con el índice más probable para cada clase
            argmax = np.array(probs).argmax(axis=0)

            #Recoge de la matriz de clases las clases correspondientes a su índice
            r = np.take(self.clases, argmax)
            return r
            
        except:
            raise ClasificadorNoEntrenado()