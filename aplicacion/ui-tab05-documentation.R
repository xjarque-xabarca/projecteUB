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



tabPanel("Documentación",
         fluidRow(
           column(4,wellPanel(
             h4("Documentación"),
             a("Entrada de datos", href = "#inputdata"), br(),
             a("Formato de los datos", href = "#dataformat"), br(),
             a("Guardar datos", href="#rdata"), br(),
             a("Visualizaciones", href="#vis"), br(),
             a("PCA Plots", href="#pcaplots"), br(),
             a("Analysis Plots", href="#analysisplots"), br(),
             a("Volcano Plots", href="#volcano"), br(),
             a("Scatterplots", href="#scatterplots"), br(),
             a("Gene Expression Boxplots", href="#boxplots"), br(),
             a("Heatmaps", href="#heatmaps"), br()
           )
           ),#column
           column(8,
                  includeMarkdown("doc/instructions.md"))
         ))