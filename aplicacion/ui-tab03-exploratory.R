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


tabPanel("Análisis exploratorio (EDA)",  
     fluidRow(
         column(4,wellPanel(
           uiOutput("campSelector")
             )
         )
         column(8,
			 tabsetPanel(id="groupplot_tabset",
						tabPanel(title="Diagrama de barras", plotOutput("barplot")
						),#tabPanel
						tabPanel(title="Resumen", verbatimTextOutput("summary")
						)#tabPanel
			 )
         )
     )
)


