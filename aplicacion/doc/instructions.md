---
title: "<b>Aplicación web con Shiny para la comparación entre métodos de aprendizaje supervisado.</b><br><br>Anexo I - Manual de usuario"
author: "Javier Jarque / Xavier Abarca"
date: "31 de julio de 2018"
output:
  html_document: default
  word_document: default
  pdf_document: default
---

**Descripción:** *Esta aplicación permite aplicar algoritmos de aprendizaje automático a un fichero definido por el usuario.*

## Instrucciones de uso

### <a name="loading"></a> Carga de datos

La primera acción a realizar deberá ser la carga de los datos, para ello ha de seleccionarse la opción de menú correspondiente:

<img src="inputdata-menucargadatos.png" alt="Carga de datos"/>

Aparecerá un formulario donde se mostrará por defecto un conjunto de datos de ejemplo (Iris dataset). Marcando la opción "Cargar fichero de datos podrá importarse un fichero con datos definidos por el usuario. En la carpeta */data* se han incluido algunos ficheros de datos para pruebas.

<img src="inputdata-checkcargardatos.png" alt="Opciones de carga de datos"/>

El fichero a importar deberá tener el formato que se describe a continuación:

**Formato de los datos**

- Debe ser un fichero .CSV utilizando como separador entre los campos el *punto y coma*.
- La primera fila debe contener los nombres de las variables.
- La última columna debe contener las clases y el resto de columnas las características.

Una vez importado en fichero, se motrarà una tabla con el contenido del mismo. La tabla mostrada tiene la siguientes características:

- Los datos por defecto se muestran paginados. 
- Puede seleccionarse el número de filas por página mediante la opción *Show ... entries *
- Pueden hacerse búsquedas mediante la opción *Search*

<img src="inputdata-compruebe.png" alt="" style="width: 100%"/>

También es posible obetener un resumen estadístico de los datos (summary), donde pueden observarse diferentes valores estadísticos de las columnas correspondientes a las *características*:

- Valor mínimo
- Primer cuartil
- Mediana
- Media
- Tercer cuartil
- Valor máximo

Además de un recuento de número de elementos de cada una de las *clases*.

<img src="inputdata-summary.png" alt="Resumen estadístico de los datos" />

<br>

### <a name="eda"></a> Análisis exploratorio (EDA)


Accediendo a la opción de menú *Análisis exploratorio (EDA)*

<img src="exploratory-menu.png" alt="" />

incialmente se muestra un diagrama de barras con el número de elementos de cada una de las clases del conjunto de datos:

<img src="exploratory-diagramabarrasclases.png" alt="" />

Mediante el desplegable que se muestra en la parte izquierda de la pantalla, es posible seleccionar también las columnas correspondientes a las características:

<img src="exploratory-seleccionevariable.png" alt="" />

y obtener un histograma de frecuencia, donse se puede observar si la distribición se aproxima a la normal.

<img src="exploratory-histogramacaract.png" alt="" />


### <a name="learning"></a> Aprendizaje automático

Accediendo a la opción de menú *Aprendizaje automático*

<img src="machinelearning-menu.png" alt="" />

se muestra un listado de algoritmos de aprendizaje de tipo supervisado, donde puede seleccionarse uno o varios de ellos y aplicarlos al conjunto de datos 
que se ha cargado previamente

<img src="machinelearning-algoritmos.png" alt="" />

Una vez seleccionados los métodos de aprendizaje, se calcularan los resultados y se mostraran en una tabla con 
los valores siguientes:

<img src="machinelearning-comparacionresultados.png" alt="" />

Para obtener los valores de accuracy para cada modelo, se hace un cross-validation de 5 splits y se obtiene el accuracy ponderado con los datos de entreno.

- Model: modelo/algoritmo de Machine Learning supervisado
- Mean.Val.Accuracy: Valor ponderado del accuracy obtenido en la cross-validation de entreno
- Str.Val.Accuracy: Desviación estándard de los valors accuracy obtenidos en el cross-validation de entreno
- min: Mínimo valor de accuracy obtenido en el entreno
- max: Máximo valor de accuracy obtenido en el entreno
- Test set Accuracy: Valor de accuracy obtenido al predecir con los datos de test (usando el modelo entrenado).

También se pueden visualizar los resultados de forma gráfica, donde se pueden observar los métodos de aprendizaje
ordenados de forma descencente por resultado de su "Accurancy".

<img src="machinelearning-plotresultados.png" alt="" />

### <a name="help"></a>Información de la aplicación

Para obtener información más detallada acceda a la pestaña "Documentación"

<img src="documentation-menu.png" alt="Documentación" />









