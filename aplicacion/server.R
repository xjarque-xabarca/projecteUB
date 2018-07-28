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

# 
options(shiny.maxRequestSize = 100*1024^2)

# Cargamos las librerías
source("load-libraries.R")
print(sessionInfo())

shinyServer(function(input, output,session) {
  ## Server functions are divided by tab
  
  ## =========================================================================== ##
  ## TAB INICIO
  ## =========================================================================== ##     
  source("server-tab01-getting-started.R",local = TRUE)
  ## =========================================================================== ##
  ## TAB CARGAR FICHERO DE DATOS
  ## =========================================================================== ##
  source("server-tab02-inputdata.R",local = TRUE)
  ## =========================================================================== ##
  ## TAB ANALISIS EXPLORATORIO
  ## =========================================================================== ##
  source("server-tab03-exploratory.R",local = TRUE)
  ## =========================================================================== ##
  ## TAB ALGORITMOS DE APRENDIZAJE 
  ## =========================================================================== ##
  source("server-tab04-machinelearning.R",local = TRUE)
  ## =========================================================================== ##
  ## TAB INSTRUCCIONES 
  ## =========================================================================== ##        
  source("server-tab05-documentation.R",local = TRUE)
  
  

  
  
  
  
  
  
  
  
  
  
  
})