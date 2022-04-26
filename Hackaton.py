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
st.set_page_config(layout="wide")
mt1, logo, mt2 = st.columns([5,6,1])

with mt1:
    st.write("")

with logo:
    st.image("logo.jpeg", width=240)

with mt2:
    st.write("")

st.header("Obtencion de Reacciones Quimicas.")

#Funcion principal del bot
def mainbot():
    try:
        while True:  

            #col1, col2, col3, col4 = st.columns([2,2,2,4]) 
            cols= st.columns([2,1,2,1,2,1,4]) 


            elemento=cols[0].selectbox("Ingrese el elemento: ", ("Elemento 1","aluminio", "bario", "berilio", "bismuto", "cadmio","calcio", "cerio", "cromo", "cobalto", "cobre", "oro",
                    "iridio", "hierro", "plomo", "litio", "magnesio", "manganeso", "mercurio", "molibdeno", "níquel", "osmio", "paladio",
                    "platino", "potasio", "radio", "rodio", "plata", "sodio", "tantalio", "talio", "torio", "estaño", "titanio", "volframio",
                    "uranio", "vanadio","cinc","escandio","titanio","vanadio","cromo","manganeso","hierro","cobalto","niquel","cobre","zinc",
                    "itrio","circonio","niobio","molibdeno","tecnecio","rutenio","rodio","paladio","plata","cadmio","lantano","hafnio","tantalio",
                    "tungsteno","renio","osmio","iridio","platino","oro","mercurio","actinio","rutherfordio","dubnium","seaborgio","bohrium",
                    "hassium","meitnerio","darmstadtium","roentgenio","copernico","cerio","praseodimio","neodimio","prometeo","samario",
                    "europio","gadolinio","terbio","disprosio","holmio","erbio","tulio","iterbio","lutecio","torio","protactinio","uranio",
                    "neptunio","plutonio","americio","curio","berkelio","californio","einstenio","fermio","mendelevio","nobelio","lawrencium",
                    "carbon","nitrogeno","fosforo","azufre","selenio","fluor","cloro","bromo","yodo","astato","tennessine"), key = "optelement1")
            prefijo1=cols[1].selectbox("Valencia 1", (-3,-2,-1,1,2,3,4,5,6,7,8,9), key= "_prefijo1")
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

            elemento2= cols[2].selectbox("Ingrese el elemento:", ("Elemento 2","hidrogeno", "oxigeno", "carbon","nitrogeno","fosforo","azufre","selenio","fluor","cloro","bromo","yodo","astato"), key = "optelement2")
            prefijo2=cols[3].selectbox("Valencia 2",(-3,-2,-1,1,2,3,4,5,6,7,8,9), key = "_prefijo2")
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


            elemento3= cols[4].selectbox("Ingrese el elemento:", ('Elemento 3','ninguno', 'agua'), key = "optelement3")
            prefijo3=cols[5].selectbox("Valencia 3",(-3,-2,-1,1,2,3,4,5,6,7,8,9), key = "_prefijo3")
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


            col5, col6 = st.columns([6,4])        
            col5.image('valencias.png')

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
            st.text(resultados)
            st.text(resultados2)
            st.text(resultados3)

            lhidruros=[['metal'], ['hidrogeno'],['ninguno']]
            loxidosm=[['metal'],['oxigeno'],['ninguno']]
            lhidroxidos=[['metal'],['oxigeno'],['agua']]
            lsalesbi=[['metal'],['no metal'],['ninguno']]
            lhidracidos=[['no metal'],['hidrogeno'],['ninguno']]
            lanhidridos=[['no metal'],['oxigeno'],['ninguno']]
            
            areareaccion = cols[6].text_area("Reaccion quimica.")

            if lentrada==lhidruros:

                cols[6].text("La reaccion quimica es:")
                cols[6].text(f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}")#Esta ya esta bien            

            elif lentrada==loxidosm:
                areareaccion.text("La reaccion quimica es:")
                areareaccion.text(f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}")#Esta ya esta bien 

            elif lentrada==lhidroxidos:
                cols[6].text("La reaccion quimica es:")
                cols[6].text(f"{obtsimbolo(elemento)}1OH{prefijo1}") #Esta ya esta bien


            elif lentrada==lsalesbi:
                cols[6].text("La reaccion quimica es:")
                cols[6].text(f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}") #Esta ya esta bien


            elif lentrada==lhidracidos:
                cols[6].text("La reaccion quimica es:")
                cols[6].text(f"H{prefijo3}{obtsimbolo(elemento2)}1") #Esta ya esta bien


            elif lentrada==lanhidridos:
                cols[6].text("La reaccion quimica es:")
                cols[6].text(f"O{prefijo2}{obtsimbolo(elemento2)}2")  #Esta ya esta bien

            else:
                cols[6].text("La reaccion no esta disponible")
                
    except Exception as er: 
        print("error: ", er)
mainbot()  

