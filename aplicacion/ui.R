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
## 
## 
# ui.R

source("load-libraries.R")

customHeaderPanel <- function(title,windowTitle=title){
  tagList(
    tags$head(
      tags$title(windowTitle),
      tags$link(rel="stylesheet", type="text/css", href="app.css"),
      tags$h1(a(href="http://http://www.ub.edu/biologia"))
    )
  )
}


tagList(
  tags$head(
    tags$style(HTML(" .shiny-output-error-validation {color: darkred; } ")),
    tags$style(".mybuttonclass{background-color:#CD0000;} .mybuttonclass{color: #fff;} .mybuttonclass{border-color: #9E0000;}")
  ),
  navbarPage(
    
    theme = "bootstrap.min.united.updated.css",
    # Theme from http://bootswatch.com/
    # title = "Meaching learing - Herramienta para la elección de algoritmos !!! ",
    title = "",
    ## TAB INICIO
    ## =========================================================================== ##    
    source("ui-tab01-getting-started.R",local=TRUE)$value,    
    ## =========================================================================== ##
    ## TAB CARGAR FICHERO DE DATOS
    ## =========================================================================== ##
    source("ui-tab02-inputdata.R",local=TRUE)$value,    
    ## =========================================================================== ##
    ## TAB ANALISIS EXPLORATORIO
    ## =========================================================================== ##
    source("ui-tab03-exploratory.R",local=TRUE)$value,
    ## =========================================================================== ##
    ## TAB ALGORITMOS DE APRENDIZAJE 
    ## =========================================================================== ##
    source("ui-tab04-machinelearning.R",local=TRUE)$value,    
    ## =========================================================================== ##
    ## TAB INSTRUCCIONES 
    ## =========================================================================== ##        
    source("ui-tab05-documentation.R",local=TRUE)$value,    
    
    ## ==================================================================================== ##
    ## FOOTER
    ## ==================================================================================== ##              
    footer=p(hr(),p("ShinyApp creada por ", strong("Javier Jarque Valentín / Xavier Abarca García"),"  ",align="center",width=4),
             p(("CURSO DE DATA SCIENCE (CIENCIA DE LOS DATOS) APLICACIONES A LA BIOLOGÍA Y A LA MEDICINA CON PYTHON Y R"),align="center",width=4),
             p(("FACULTAT DE BIOLOGIA – UNIVERSITAT DE BARCELONA"),align="center",width=4),
             p(("Código disponible en Github:"),a("https://github.com/xjarque-xabarca",href="https://github.com/xjarque-xabarca"),align="center",width=4)
    ),
    
    ## ==================================================================================== ##
    ## GOOGLE ANALYTICS
    ## ==================================================================================== ## 
    tags$head(includeScript("google-analytics.js"))
  )
)

