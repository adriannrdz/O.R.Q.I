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
mt1, header, mt2 = st.columns([5,5,3])
st.markdown("""
                    <style>
                    .big-font {
                        font-size:30px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

with mt1:
    st.text("")

with header:
    st.image("logo.jpg", width = 300)
    st.markdown('<p class="big-font"> Obtencion de Reacciones Quimicas. </p>', unsafe_allow_html=True)
    


with mt2:
    st.text("")    
   

#Funcion principal del bot
def mainbot():
    try:
        while True:  

            #col1, col2, col3, col4 = st.columns([2,2,2,4]) 
            cols= st.columns([1,1,1,1,1,1,1,4]) 


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

            elemento2= cols[2].selectbox("Ingrese el elemento:", ("Elemento 2","hidrogeno", "oxigeno", "carbono","nitrogeno","fosforo","azufre","selenio","fluor","cloro","bromo","yodo","astato"), key = "optelement2")
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



            lhidruros=[['metal'], ['hidrogeno'],['ninguno']]
            loxidosm=[['metal'],['oxigeno'],['ninguno']]
            lhidroxidos=[['metal'],['oxigeno'],['agua']]
            lsalesbi=[['metal'],['no metal'],['ninguno']]
            lhidracidos=[['no metal'],['hidrogeno'],['ninguno']]
            lanhidridos=[['no metal'],['oxigeno'],['ninguno']]
            
            cols[6].text("")
            cols[6].image("arrow.jpg")
            
            st.markdown("""
                    <style>
                    .big-font2 {
                        font-size:45px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
            
            

            if lentrada==lhidruros:

                #cols[6].text(f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}")
                rhidruro = (f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}")
                cols[7].text("")
                cols[7].markdown(f'<p class="big-font2">{rhidruro}</p>', unsafe_allow_html=True)
                st.subheader("Hidruros:")
                st.text("""
                - Los hidruros de los metales alcalinos (grupo I), alcalinotérreos (grupo II) y de algunos otros elementos del sistema periódico se forman directamente de los elementos a temperaturas elevadas.
                - Otros pueden ser formados por intercambio del anión.
                - Algunos hidruros de los elementos nobles se pueden formar también aprovechando el hidrógeno del metanol.
                - Por lo general los hidruros son compuestos estequiométricos que presentan propiedades metálicas como por ejemplo la conductividad.
                - Tienen una gran velocidad de difusión del hidrógeno por medio del sólido cuando se someten a temperatura elevadas.
                - Los hidruros salinos por lo general se encuentran en estado sólido y son de color blanco o gris.
                - Este tipo de hidruro se puede obtener por medio de reacciones directas del metal con el hidrógeno.
                - Elementos como el fósforo, arsénico y sus compuestos son considerados tóxicos.
                - Tienen agentes reductores bastante efectivos y no reaccionan con el agua o con los ácidos catalogados como no oxidantes.
                """)
                st.image("hidruros.jpg")
                                       

            elif lentrada==loxidosm:
                roxido = (f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}")
                cols[7].text("")
                cols[7].markdown(f'<p class="big-font2">{roxido}</p>', unsafe_allow_html=True)
                st.subheader("Oxidos Metalicos:")
                st.text("""
                - Son combinaciones binarias de un metal con el oxígeno, en las que el oxígeno tiene número de oxidación (-2.)
                - Poseen basicidad la cual es la capacidad que tiene una sustancia de ser ácido neutralizante cuando se encuentra en una solución de tipo acuosa.
                - Tiene anfoterismo, en este aspecto se debe recordar que no todos los óxidos son iguales por lo que mientras esté ubicado más hacia la izquierda, el metal será más básico que el óxido.
                - Por lo general, tienden a ser sólidos.
                - Tienen un punto de fusión relativamente alto.
                - Son cristalinos en su gran mayoría y, además, son medianamente solubles en agua.
                - Son buenos conductores de la electricidad.
                - Pueden estar presentes en los tres diferentes estados de agregación de la materia.
                - Son compuestos de tipo inorgánico.
                - El proceso para que se obtengan depende en gran manera de la naturaleza del metal o de la reactividad y de las condiciones físicas que le rodean.
                           """)
                st.image("oxidos.jpg", width = 350)


            elif lentrada==lhidroxidos:
                rhidroxido= (f"{obtsimbolo(elemento)}1OH{prefijo1}")
                cols[7].text("")
                cols[7].markdown(f'<p class="big-font2">{rhidroxido}</p>', unsafe_allow_html=True)
                st.subheader("Hidroxidos:")
                st.text("""
                - Los hidróxidos son un tipo de compuesto químico que está formado a partir de la unión de un elemento de tipo metálico o catiónico con un elemento que pertenece al grupo de los hidróxidos, o aniones.
                - Se caracterizan por ser compuestos inorgánicos ternarios porque tienen en su molécula de hidrógeno, una molécula de oxígeno y un elemento de tipo metálico.
                - Pueden desasociarse cuando se disuelven en agua. Tienen un carácter básico bastante fuerte porque el grupo hidroxilo tiene la capacidad de captar protones.
                - Pueden hacer que el color del papel tornasol de un tono rojo a uno azul.
                - Tienen la capacidad de reaccionar con los ácidos y producir de esta manera una sal y agua.
                - Cuando hacer algún tipo de reacción puede liberar energía.
                            """)
                st.image("hidroxidos.jpg")



            elif lentrada==lsalesbi:
                rsalesbi =(f"{obtsimbolo(elemento)}{prefijo2}{obtsimbolo(elemento2)}{prefijo1}")
                cols[7].text("")
                cols[7].markdown(f'<p class="big-font2">{rsalesbi}</p>', unsafe_allow_html=True)
                st.subheader("Sales Binarias:")
                st.text("""
                - Una sal binaria es una combinación que se hace entre un metal y un no metal en su estructura y que poseen además una fórmula química general como MmXn, donde M será el elemento metálico mientras que X el no metálico.
                - Tienen cierta semejanza con la sal de cocina.
                - Son también conocidas con el nombre de sales neutras.
                - Tienen una consistencia sólida, son muy cristalinas y por lo general su color es blanco.
                - Posee un sabor salado.
                - Gran parte de las sales binarias son solubles cuando entran en contacto con el agua.
                - En las sales binarias el elemento no metálico siempre actuará con su valencia fija frente al hidrógeno.
                - Es uno de los tres tipos de sales que se pueden encontrar en la naturaleza.
                - El no metal que forma parte de las sales binarias siempre utilizará su menor valencia.
                - Son consideradas como parte fundamental de la química, principalmente en el campo educativo.
                            """)
                st.image("salesbi.jpg")



            elif lentrada==lhidracidos:
                rhidracido= (f"H{prefijo3}{obtsimbolo(elemento2)}1")
                cols[7].text("")
                cols[7].markdown(f'<p class="big-font2">{rhidracido}</p>', unsafe_allow_html=True)
                st.subheader("Sales Hidracidos:")
                st.text("""
                -También son conocidos como ácidos hidrácidos, sales binarias o hídricos ácidos, están hechos por dos compuestos binarios ácidos los cuales deben ser un hidrógeno
                (del grupo halógeno o antígeno en la tabla periódica) y un no metal, sin presencia de oxígeno.
                -Cuando se disuelven en agua son solubles.
                -No poseen oxígeno en su composición.
                -Liberan usualmente un muy denso humo blanco.
                -Cuando se encuentran en su estado natural son gaseosos y para nombrarlos se utiliza el nombre del no metal con la terminación uro y seguido del nombre Hidrógeno.
                -Se obtienen también por disolución acuosa.
                -Son perfectos cuando se utilizan como conductor de energía eléctrica.
                -Con respecto a su aspecto éstos son incoloros, son soluciones transparentes.
                -También cuando entran en contacto con agua originan soluciones ácidas.
                -Solo existen sietes ácidos hidrácidos que forman solo con los no metales de Cl, S, I, F, Se, Te.
                -Sus puntos de ebullición son superiores a la de los anhídridos, por ejemplo, el Cloruro de Hidrógeno hierve a -85Co pero el Ácido Clorhídrico hierve a -48Co.
                """)
                st.image("hidracidos.jpg")

                


            elif lentrada==lanhidridos:
                ranhidrido = (f"O{prefijo1}{obtsimbolo(elemento1)}2")
                cols[7].text("")
                cols[7].markdown(f'<p class="big-font2">{ranhidrido}</p>', unsafe_allow_html=True)
                st.subheader("Anhidridos:")
                st.text("""
                -En química podemos definir los anhídridos como un compuestos químicos de tipo binario que surgen al juntar un No Metal con Oxígeno,
                también son llamados como Óxidos No Metálicos u Óxidos Ácidos, cuya fórmula general es (RCO)2º.
                -Son productos de la condensación de 2 moléculas de ácidos carboxílicos. 
                -La fórmula de los anhídridos es del tipo X2On (donde X es un elemento no metálico y O es oxígeno) Ejemplos: SeO, So3, CO2. Poseen la característica de ser especialmente reactivos.
                -Se pueden manejar en muchas áreas, lo habitual, donde son más vistos y usados es en la fabricación de medicamentos.
                -Como por ejemplo en el ámbito medico existe uno muy usado llamado Ácido Acetilsalicílico un fármaco que es muy utilizado.
                -Como ya se ha mencionado vienen a ser fruto de la deshidratación de dos moléculas, pero también existen casos donde pudiese ser una, en el caso intramolecular, de ácido carboxílico.
                -También son llamados anhídridos carboxílicos.
                """)
                st.image("anhidridos.jpg")

               

            else:
                cols[7].text("La reaccion no esta disponible")    
                
            mt3, val, mt4 = st.columns([1,5,2])
            
            with mt3:
                st.text("")
                
            with val:
                st.subheader("Tabla de Valencias:")
                st.image('valencias.png')
                
            with mt4:
                st.text("")
                
            st.subheader("Probabilidades:")
            st.text("""
            Las cantidades que se muestran a continuacion representan las probabilidaddes que determina el programa con la finalidad de establecer
            el tipo de elemento que se asigno en la parte superior de la pagina:
            El orden de los numeros representan: (Agua, Hidroxido, Metal, No metal y Oxigeno, respectivamente)""") 
            
            st.text(lentrada)
            st.text("Primer elemento")
            st.text(resultados)
            st.text("Segundo elemento")
            st.text(resultados2)
            st.text("Tercer elemento")
            st.text(resultados3)
            
            
                
    except Exception as er: 
        print("error: ", er)
mainbot()  

