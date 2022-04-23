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
modelo.fit(entrenamiento, salida, n_epoch=2000, batch_size=50, show_metric=True)
modelo.save("modelo.tflearn")

#Funcion principal del bot
def mainbot():
    while True:         
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

        salbinaria=st.text_input("Es una sal binaria? S/N")
               
        prefijo1=st.text_input("Inserte el numero correspondiente al prefijo de acuerdo a la nomenclatura sistematica")
        elemento= st.text_input("Ingrese el elemento: ")
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
        
        prefijo2=st.text_input("Inserte el numero correspondiente al prefijo de acuerdo a la nomenclatura sistematica:")
        elemento2= st.text_input("Ingrese el segundo elemento:")
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

        time.sleep(1.5)

        st.text("Ahora seleccion los reactivos restantes")  

        time.sleep(1.5)

        prefijo3=st.text_input("Inserte el numero correspondiente al prefijo de acuerdo a la nomenclatura sistematica:  ")
        elemento3= st.text_input("Ingrese el tercer elemento:")
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

        prefijo4=st.text_input("Inserte el numero correspondiente al prefijo de acuerdo a la nomenclatura sistematica:")
        elemento4= st.text_input("Ingrese el cuarto elemento:")
        cubeta4= [0 for _ in range(len(palabras))]
        entradaprocesada4= nltk.word_tokenize(elemento4)
        entradaprocesada4 =[stemmer.stem(palabra.lower()) for palabra in entradaprocesada4]
        for palabraindividual4 in entradaprocesada4:
            for i, palabra in enumerate(palabras):
                if palabra == palabraindividual4:
                    cubeta4[i] = 1
        resultados4 = modelo.predict([numpy.array(cubeta4)])
        resultadosindices4= numpy.argmax(resultados4)
        tag4=tags[resultadosindices4]

        for tagaux in datos["contenido"]:
            if tagaux["tag"]== tag4:
                respuesta4 = tagaux["respuesta"] 


        #Apartado para la sales binarias
        if salbinaria == 'S':

            prefijo5=st.text_input("Inserte el numero correspondiente al prefijo de acuerdo a la nomenclatura sistematica:")
            elemento5= st.text_input("Ingrese el quinto elemento:")
            cubeta5= [0 for _ in range(len(palabras))]
            entradaprocesada5= nltk.word_tokenize(elemento5)
            entradaprocesada5 =[stemmer.stem(palabra.lower()) for palabra in entradaprocesada5]
            for palabraindividual5 in entradaprocesada5:
                for i, palabra in enumerate(palabras):
                    if palabra == palabraindividual5:
                        cubeta5[i] = 1
            resultados5 = modelo.predict([numpy.array(cubeta5)])
            resultadosindices5= numpy.argmax(resultados5)
            tag5=tags[resultadosindices5]

            for tagaux in datos["contenido"]:
                if tagaux["tag"]== tag5:
                    respuesta5 = tagaux["respuesta"] 
        

        st.text(resultados)        
        st.text("BOT: ",(respuesta))
        st.text(resultados2)
        st.text("BOT: ",(respuesta2))
        st.text(resultados3)
        st.text("BOT: ",(respuesta3))
        st.text(resultados4)
        st.text("BOT: ",(respuesta4))
        if salbinaria=='S':
            st.text(resultados5)
            st.text(respuesta5)
        
        st.text(prefijo1)
        st.text(prefijo2)   
        st.text(prefijo3)   
        st.text(prefijo4) 
        if salbinaria == 'S':
            st.text(prefijo5) 
        
        def obtsimbolo(simb):
            simboloq=df[df['nombre']==simb]['simbolo']
            return simboloq
         

        #Reglas de nomenclatura sistematica
        
        #Lista para comparar con las listas preestablecidas
        lentrada=[]
        for i in respuesta:
            lentrada.append(respuesta)
        
        for i in respuesta2:
            lentrada.append(respuesta2)
        
        for i in respuesta3:
            lentrada.append(respuesta3)
        
        for i in respuesta4:
            lentrada.append(respuesta4)

    
        print(lentrada)
        
        lhidruros=[['metal'],['ninguno'], ['hidrogeno'],['ninguno']]
        loxidosm=[['metal'],['ninguno'],['oxigeno'],['ninguno']]
        lhidroxidos=[['metal'],['oxigeno'],['agua'],['ninguno']]
        lsalesbi=[['metal'],['ninguno'],['no metal'],['ninguno']]
        lhidracidos=[['hidrogeno'],['ninguno'],['no metal'],['ninguno']]
        lanhidridos=[['oxigeno'],['ninguno'],['no metal'],['ninguno']]
        """
        lacidosoxi=[['oxigeno'],['no metal'],['agua'],['ninguno']]
        lsalesoxi=[['metal'],['hidroxido'],['oxigeno'],['no metal'],['agua']]  
        """

    
        
        if lentrada==lhidruros:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"{obtsimbolo(elemento)}{prefijo3}{obtsimbolo(elemento3)}{prefijo1}")#Esta ya esta bien            
        
        elif lentrada==loxidosm:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"{obtsimbolo(elemento)}{prefijo3}{obtsimbolo(elemento3)}{prefijo1}")#Esta ya esta bien 

        elif lentrada==lhidroxidos:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"{obtsimbolo(elemento)}1OH{prefijo1}") #Esta ya esta bien
            

        elif lentrada==lsalesbi:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"{obtsimbolo(elemento)}{prefijo3}{obtsimbolo(elemento3)}{prefijo1}") #Esta ya esta bien
             

        elif lentrada==lhidracidos:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"H{prefijo3}{obtsimbolo(elemento3)}1") #Esta ya esta bien
            

        elif lentrada==lanhidridos:
            st.text("La reaccion quimica es:")
            time.sleep(1.5)
            st.text(f"O{prefijo3}{obtsimbolo(elemento3)}2")  #Esta ya esta bien

        #Checar si agregamos estas (opcionales)
        """        
        elif lentrada==lacidosoxi:
            print("La reaccion quimica es:")
            time.sleep(1.5)
            print(f"H{}")          

        elif lentrada==lsalesoxi:
            print("La reaccion quimica es:")
            time.sleep(1.5)
            print(f"{obtsimbolo(elemento)}{prefijo3}{obtsimbolo(elemento3)}{prefijo1}") 
        """    

mainbot()  

