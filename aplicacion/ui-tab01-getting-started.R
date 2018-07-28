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


tabPanel("Inicio",
         fluidRow(
           column(4,wellPanel(
             h4("Inicio"),
             a("Carga del fichero de datos (Dataset)", href="#loading"),br(),
             a("Análisis exploratorio (EDA)", href = "#exploratory"), br(),
             a("Aprendizaje automático", href="#learning"), br(),
             a("Documentación", href = "#help"), br()
           )
           ),#column
           column(8, includeMarkdown("doc/instructions.md"))
         ))



