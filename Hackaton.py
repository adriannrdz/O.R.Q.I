import streamlit as st
import nltk 
from nltk.stem.lancaster import LancasterStemmer 
stemmer = LancasterStemmer()
import tflearn 
import numpy
import tensorflow 
import json
import random
import pickle
import time
import openpyxl
import pandas as pd


datosexc=pd.read_excel('listaelem.xlsx')
df=pd.DataFrame(datosexc)

    
nltk.download('punkt')

with open("contenido.json", encoding='utf-8') as archivo: 
    datos =  json.load(archivo)
  

try:
    with open("variables.pickle", "rb") as archivopickle:
        palabras, tags, entrenamiento, salida = pickle.load(archivopickle)
except:
    palabras=[]
    tags= []
    auxX=[]
    auxY=[]

    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
            auxpalabra = nltk.word_tokenize(patrones)
            palabras.extend(auxpalabra)
            auxX.append(auxpalabra)
            auxY.append(contenido["tag"])

            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

    palabras =[stemmer.stem(w.lower()) for w in palabras if w!="?"]
    palabras = sorted(list(set(palabras)))
    tags= sorted(tags)

    entrenamiento = []
    salida =[]

    salidavacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX): 
        cubeta = []
        auxpalabra= [stemmer.stem(w.lower())for w in documento]
        for w in palabras:
            if w in auxpalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filasalida= salidavacia[:]
        filasalida [tags.index(auxY[x])]= 1
        entrenamiento.append(cubeta)
        salida.append(filasalida)

    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)
    with open("variables.pickle","wb") as archivopickle:
        pickle.dump((palabras, tags, entrenamiento, salida),archivopickle)

#Entra la libreria Tf para el machine learning
tensorflow.compat.v1.reset_default_graph()

red = tflearn.input_data(shape=[None, len(entrenamiento[0])])
red = tflearn.fully_connected(red, 100)
red= tflearn.fully_connected(red, 100)
red= tflearn.fully_connected(red, len(salida[0]), activation="softmax")
red = tflearn.regression(red)

modelo = tflearn.DNN(red)
modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=50, show_metric=True)
modelo.save("modelo.tflearn")

#Apartado de instrucciones.
st.text("""   Buen dia, en el presente programa el usuario tendra la posibilidad de formular reacciones quimicas
    1.- Si lo que intenta ingresar es un elemento de la tabla periodica, en los apartados de prefijos coloque la opcion ninguno.
    2.- Si lo que desea agregar es un compuesto, por favor ingrese dicho compuestos de acuerdo a la nomenclatura sistematica.
    Ejemplo. 
    Compuesto a ingresar: Dioxido de Titanio. (TiO2)
    Prefijo 1= como es di colocamos el numero 2
    Elemento 1= oxigeno
    Prefijo 2= como es mono, colocamos el 1
    Elemento 2= Titanio """)

#Funcion principal del bot
def mainbot():
    while True:  
        quest1=1
        quest2 = 2
        quest3 = 3
        quest4= 4
        quest5 = 5
        quest6 = 6
                       
        prefijo1=st.text_input("Valencia 3", key =quest1)
        elemento= st.text_input("Ingrese el elemento: ", key=quest2)
        cubeta= [0 for _ in range(len(palabras))]
        entradaprocesada= nltk.word_tokenize(elemento)
        entradaprocesada =[stemmer.stem(palabra.lower()) for palabra in entradaprocesada]
        for palabraindividual in entradaprocesada:
            for i, palabra in enumerate(palabras):
                if palabra == palabraindividual:
                    cubeta[i] = 1
        resultados = modelo.predict([numpy.array(cubeta)])
        resultadosindices= numpy.argmax(resultados)
        tag=tags[resultadosindices]

        for tagaux in datos["contenido"]:
            if tagaux["tag"]== tag:
                respuesta = tagaux["respuesta"]
        
        prefijo2=st.text_input("Valencia 2:", key=quest3)
        elemento2= st.text_input("Ingrese el segundo elemento:", key=quest4)
        cubeta2= [0 for _ in range(len(palabras))]
        entradaprocesada2= nltk.word_tokenize(elemento2)
        entradaprocesada2 =[stemmer.stem(palabra.lower()) for palabra in entradaprocesada2]
        for palabraindividual2 in entradaprocesada2:
            for i, palabra in enumerate(palabras):
                if palabra == palabraindividual2:
                    cubeta2[i] = 1
        resultados2 = modelo.predict([numpy.array(cubeta2)])
        resultadosindices2= numpy.argmax(resultados2)
        tag2=tags[resultadosindices2]

        for tagaux in datos["contenido"]:
            if tagaux["tag"]== tag2:
                respuesta2 = tagaux["respuesta"]

        st.text("Ahora seleccion los reactivos restantes")

        prefijo3=st.text_input("Valencia 3:  ", key=quest5)
        elemento3= st.text_input("Ingrese el tercer elemento:", key=quest6)
        cubeta3= [0 for _ in range(len(palabras))]
        entradaprocesada3= nltk.word_tokenize(elemento3)
        entradaprocesada3 =[stemmer.stem(palabra.lower()) for palabra in entradaprocesada3]
        for palabraindividual3 in entradaprocesada3:
            for i, palabra in enumerate(palabras):
                if palabra == palabraindividual3:
                    cubeta3[i] = 1
        resultados3 = modelo.predict([numpy.array(cubeta3)])
        resultadosindices3= numpy.argmax(resultados3)
        tag3=tags[resultadosindices3]

        for tagaux in datos["contenido"]:
            if tagaux["tag"]== tag3:
                respuesta3 = tagaux["respuesta"] 
        
        def obtsimbolo(simb):
           # simboloq=df[df['nombre']==simb]['simbolo']
            dfSearchedElement = df['simbolo'].where(df['nombre']==simb)
            dfSearchedElement.dropna(inplace=True)
            return dfSearchedElement.squeeze()
          
           # return simboloq
         
        lentrada=[]
        for i in respuesta:
            lentrada.append(respuesta)
        
        for i in respuesta2:
            lentrada.append(respuesta2)
        
        for i in respuesta3:
            lentrada.append(respuesta3)
            
        st.text(lentrada)
        
        lhidruros=[['metal'], ['hidrogeno'],['ninguno']]
        loxidosm=[['metal'],['oxigeno'],['ninguno']]
        lhidroxidos=[['metal'],['oxigeno'],['agua'],['ninguno']]
        lsalesbi=[['metal'],['no metal'],['ninguno']]
        lhidracidos=[['hidrogeno'],['no metal'],['ninguno']]
        lanhidridos=[['oxigeno'],['no metal'],['ninguno']]
            
        
        if lentrada==lhidruros:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}")#Esta ya esta bien            
        
        elif lentrada==loxidosm:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}")#Esta ya esta bien 

        elif lentrada==lhidroxidos:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"{obtsimbolo(elemento)}1OH{prefijo1}") #Esta ya esta bien
            

        elif lentrada==lsalesbi:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}") #Esta ya esta bien
             

        elif lentrada==lhidracidos:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"H{prefijo3}{obtsimbolo(elemento2)}1") #Esta ya esta bien
            

        elif lentrada==lanhidridos:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"O{prefijo2}{obtsimbolo(elemento2)}2")  #Esta ya esta bien

mainbot()  

