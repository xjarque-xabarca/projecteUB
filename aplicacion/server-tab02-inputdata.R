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


observe({
  # Check if example selected, or if not then ask to upload a file.
  validate(
    need((input$data_file_type=="examplecounts")|((!is.null(input$rdatafile))|(!is.null(input$datafile))), 
         message = "Seleccione un fichero ... ")
  )
  inFile <- input$datafile
  if(!is.null(inFile)) {
    # update options for various analyzed data columns
  }
  
})


inputDataReactive <- reactive({
  
  # Check if example selected, or if not then ask to upload a file.
  validate(
    need((input$data_file_type=="examplecounts")|((!is.null(input$rdatafile))|(!is.null(input$datafile))), 
         message = "Seleccione un fichero ... ")
  )
  inFile <- input$datafile
  
  if(input$data_file_type=="examplecounts") {
    # example dataset
    mydataset <- iris 
    print("example dataset")
    return(list('data'=mydataset))
    
  }else { # if uploading data
    if (!is.null(inFile)) {
      
      print('uploaded dataset ......')      
      mydataset <- loadCSV(inFile$datapath)
      validate(need(ncol(mydataset)>1, message="El fichero no parece tener un formato correcto. 
      Compruebe que se trata de un fichero de tipo .csv con punto y coma como separador de las columnas."))
      return(list('data'=mydataset))
    } else{
      return(NULL)
    }
  }
})


# check if a file has been uploaded and create output variable to report this
output$fileUploaded <- reactive({
  return(!is.null(inputDataReactive()))
})

outputOptions(output, 'fileUploaded', suspendWhenHidden=FALSE)


output$dataDT <- renderDataTable({
  tmp <- inputDataReactive()
  if(!is.null(tmp)) tmp$data
})

observeEvent(input$upload_data, ({
  updateCollapse(session,id =  "input_collapse_panel", open="analysis_panel",
                 style = list("analysis_panel" = "success",
                              "data_panel"="primary"))
}))

observeEvent(inputDataReactive(),({
  updateCollapse(session,id =  "input_collapse_panel", open="data_panel",
                 style = list("analysis_panel" = "default",
                              "data_panel"="success"))
})
)

output$exploratory <- renderPrint({
  mydataset <- inputDataReactive()$data
  summary(mydataset)
})









