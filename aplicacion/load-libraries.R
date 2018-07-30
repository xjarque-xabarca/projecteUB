## ============================================================================================================ ##
# Aplicación Shiny para la comparación entre métodos de aprendizaje supervisado.
# 
# CURSO DE DATA SCIENCE (CIENCIA DE LOS DATOS) APLICACIONES A LA BIOLOGÍA Y A LA MEDICINA CON PYTHON Y R
# 
# FACULTAT DE BIOLOGIA – UNIVERSITAT DE BARCELONA
#
# Creada por Javier Jarque Valentín y Xavier Abarca García
# 
# Para contactar con los autores de este código envía un correo a <javier.jarque@gmail.com> o <xabarca@gmail.com>
# 
# Código disponible en Github: https://github.com/xjarque-xabarca
# 
## ============================================================================================================ ##


## =========================================================================== ##
## LOADING LIBRARIES
## =========================================================================== ##  

library(shiny) #devtools::install_github("rstudio/shiny"); devtools::install_github("rstudio/shinyapps")
library(gplots)
library(ggplot2)
library(shinyBS)
library(markdown)


## =========================================================================== ##
## CONFIGURING PYTHON FROM R
## =========================================================================== ##     


source("python/load-python-libraries.R",local = TRUE)


