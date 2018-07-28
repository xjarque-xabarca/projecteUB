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


tabPanel("Aprendizaje automático",  
     fluidRow(
         column(4,wellPanel(
           h4("Selecciona los algoritmos de aprendizaje automático:"),
           br(),
           checkboxInput('LinearSVC', 'LinearSVC'),           
           checkboxInput('KNeighbors', 'KNeighbors'),
           checkboxInput('RFBagging', 'RFBagging'),
           checkboxInput('RandomForest', 'RandomForest'),
           checkboxInput('AdaBoost', 'AdaBoost'),
           checkboxInput('DecisionTree', 'DecisionTree'),
           checkboxInput('ExtraTrees', 'ExtraTrees'),
           checkboxInput('Ridge', 'Ridge'),
           checkboxInput('SGD', 'SGD'),
             )
         ),
         column(8,
                tabsetPanel(id="groupplot_tabset",
                            
                            tabPanel(title="Comparación de resultados",
                                     h4("Resultados"),
                                     tableOutput("evaluate")
                            ),
                            tabPanel(title="Comparacón gráfica", plotOutput("plot")                            
                            )
                                     
                            
                            #,#tabPanel
                            
                            #tabPanel(title="Sample Distance Heatmap",
                            #         h4("Observations"),
                            #         verbatimTextOutput("KNeighborsClassifier"),
                            #         verbatimTextOutput("LinearSVC"),
                            #         verbatimTextOutput("BaggingClassifier"),
                            #         verbatimTextOutput("RandomForestClassifier")
                            #)#tabPanel
               )
          )
     )
)
