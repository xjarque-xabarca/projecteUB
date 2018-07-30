---
title: "Aplicación web con Shiny para la comparación entre métodos de aprendizaje supervisado."
author: "Javier Jarque / Xavier Abarca"
date: "31 de julio de 2018"
output:
  html_document: default
  pdf_document: default
  word_document: default
editor_options: 
  chunk_output_type: inline
---


## Documentación

Esta aplicación ha sido desarrollada por Javier Jarque Valentín y Xavier Abarca García como trabajo final del CURSO DE DATA SCIENCE (CIENCIA DE LOS DATOS) APLICACIONES A LA BIOLOGÍA Y A LA MEDICINA CON PYTHON Y R realizado por la FACULTAT DE BIOLOGIA de la  UNIVERSITAT DE BARCELONA.

[Descargar la memoria del proyecto]("proyecto_ub_ciencia_de_datos_xjarque_xabarca.pdf")

### <a name="instalation"></a>Instrucciones de instalación

**Pasos a seguir**




A continuación se muestra los pasos a seguir para instalar esta aplicacíón. 

- Las instrucciones hacen referencia a la instalación en un sistema operativo de tipo Linux, en concreto en la distribución Lubuntu:

<img src="install-lubuntu-desktop.png" alt="" width="100%" />

<br>




- Descargar *R Studio*. En las imágenes siguientes pueden verse las versiones de R y RStudio utilizadas para la elaboración de este manual:

https://www.rstudio.com/products/rstudio/download/

<img src="install-rstudio-home.png" alt="" width="100%" />

<br>

<img src="install-rstudio-version.png" alt="" width="100%" />

<br><br>


- Descargar el código fuente del proyecto de GitHub:

```
git clone https://github.com/xjarque-xabarca/projecteUB
```



<img src="install-git-clone.png" alt="" width="100%" />

<br>




- Cargar el proyecto en RStudio:

<img src="install-rstudio-open-projecteUB.png" alt="" width="100%" />

<br>




- Descargar Python (miniconda) del siguiente enlace:

<a href="https://conda.io/miniconda.html">Descargar miniconda</a>

<img src="install-download-miniconda.png" alt="" width="100%" />

<br>





- Instalar miniconda ejecutando el fichero *Miniconda3-latest-Linux-x86.sh*

<img src="install-install-miniconda.png" alt="" width="100%"  />

<br>


- Instalar las librerías de Python necesarias:

```
datascience@dspc:~/miniconda3/bin$ ./conda install pandas scikit-learn
```




- Configurar *Reticulate* para poder ejecutar Python. Abrir el fichero *doc/load-python-libraries.R* e indicar la ruta donde está instalado Python en *use_python*  :

<img src="install-configure-python.png" alt="" width="100%" />

<br>






- Instalar las librerias de R necesarias para ejecutar el proyecto. Fichero *install.R*:

<img src="install-install-libraries.png" alt="" width="100%" />

<br>





- ejecución de la aplicación y captura general












**Decargar máquina virtual para Oracle Virtual Box**

- Está disponible un máquina virtual con los fuentes de la aplicación y el software necesario para ejectuarla en la siguiente dirección:

[Descargar la máquina virtual con el proyecto instalado]("drive")

Una vez descargada importela desde *Oracle virtual box*

<img src="install-lubuntu-virtualbox.png" alt="" width="100%" />

Configure la *memoria base* del sistema con **4GB** para su correcto funcionamiento:

<img src="install-lubuntu-sistema.png" alt="" width="100%" />




















