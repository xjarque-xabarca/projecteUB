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



output$campSelector <- renderUI({
  mydataset <- inputDataReactive()$data
  colnames <- names(mydataset)
  listcolnames <- as.list(colnames)
  selectInput("variables", "Seleccione una variable:", as.list(colnames), selected=colnames[length(colnames)]) 
})


output$summary <- renderPrint({
  mydataset <- inputDataReactive()$data
  summary(mydataset)
})


output$barplot <- renderPlot({
  
  mydataset <- inputDataReactive()$data
  colnames <- names(mydataset)
  lastcolname <-colnames[length(colnames)]
  name <- input$variables

  print(name)
  print(lastcolname)

  if (name == lastcolname) {
    
    myclasses <- mydataset[ncol(mydataset)]
    myclassesnames <- colnames(mydataset)[ncol(mydataset)]
    mytitle <- paste(c("Diagrama de barras de" , myclassesnames, collapse=" "))
    barplot(table(myclasses), col="blue", xlab=myclassesnames, ylab="Número de elementos", main=mytitle)    
    
  } else {
  
    listdata <- mydataset[name]
    elem <- listdata[,1]
    
    # add a normal distribution line in histogram
    hist(elem, freq=FALSE, col="gray", xlab=name, main="Colored histogram")
    curve(dnorm(x, mean=mean(elem), sd=sd(elem)), add=TRUE, col="red") #line
    
  } 

})


