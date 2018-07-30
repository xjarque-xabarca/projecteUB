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


tabPanel("Carga de datos", 
         fluidRow(column(4,
           wellPanel(
             radioButtons('data_file_type','Utilice los datos de ejemplo o cargue su propio fichero de datos',
                          c('Datos de ejemplo'="examplecounts",
                            'Cargar fichero de datos'="upload"
                          ),selected = "examplecounts"),
             
             conditionalPanel(condition="input.data_file_type=='upload'",
                              fileInput('datafile', 'Seleccione el fichero que contiene los datos (.CSV)',
                                        accept=c('text/csv', 
                                                 'text/comma-separated-values,text/plain', 
                                                 '.csv'))
             )
             #,
             #conditionalPanel("output.fileUploaded",
             #                actionButton("upload_data","Submit Data",
             #                             style="color: #fff; background-color: #CD0000; border-color: #9E0000"))
             
            )
         ),
         column(8,
                bsCollapse(id="input_collapse_panel",open="data_panel",multiple = FALSE,
                           bsCollapsePanel(title="Compruebe los datos antes  de continuar",value="data_panel",
                                           dataTableOutput('dataDT')                       
                           ),
                           bsCollapsePanel(title="Resumen estatístico de los datos",value="analysis_panel",
                                           verbatimTextOutput('exploratory'),
                                           tags$head(tags$style(".mybuttonclass{background-color:#CD0000;} .mybuttonclass{color: #fff;} .mybuttonclass{border-color: #9E0000;}"))
                           )
                )#bscollapse
         )#column
         )#fluidrow
)#tabpanel
