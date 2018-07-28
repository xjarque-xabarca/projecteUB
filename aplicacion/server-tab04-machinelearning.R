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



output$analysisoutput4 <- renderPlot({
  d <- iris
  plot(d$Sepal.Width, d$Sepal.Length)
})


classifiersReactive <- reactive({
	
  print("Executing classifiersReactive ...")
  mydataset <- inputDataReactive()$data  
  
  classifiers <- c("")
  
  if (input$LinearSVC) {
    print("Has seleccionado el algoritmo: Linear SVC ")
    classifiers <- c(classifiers, 'Linear SVC')
  }  
  
  if (input$KNeighbors) {
    print("Has seleccionado el algoritmo: KNeighbors ")
    classifiers <- c(classifiers, 'KNeighbors')
  }
  
  if (input$RFBagging) {
    print("Has seleccionado el algoritmo: RF-Bagging ")
    classifiers <- c(classifiers, 'RF-Bagging')
  }
  
  if (input$RandomForest) {
    print("Has seleccionado el algoritmo: RandomForest ")
    classifiers <- c(classifiers, 'RandomForest')
  }  
  
  if (input$AdaBoost) {
    print("Has seleccionado el algoritmo: AdaBoost ")
    classifiers <- c(classifiers, 'AdaBoost')
  }    
  
  if (input$DecisionTree) {
    print("Has seleccionado el algoritmo: DecisionTree ")
    classifiers <- c(classifiers, 'DecisionTree')
  }    
  
  if (input$ExtraTrees) {
    print("Has seleccionado el algoritmo: ExtraTrees ")
    classifiers <- c(classifiers, 'ExtraTrees')
  }    
  
  if (input$Ridge) {
    print("Has seleccionado el algoritmo: Ridge ")
    classifiers <- c(classifiers, 'Ridge')
  }      
  
  if (input$SGD) {
    print("Has seleccionado el algoritmo: SGD ")
    classifiers <- c(classifiers, 'SGD')
  }        
  
  print(classifiers)
  
  # classifiers = c( "Linear SVC",  "KNeighbors", "RF-Bagging", "RandomForest", "AdaBoost", "DecisionTree", "ExtraTrees", "Ridge", "SGD" )  
  
  scores <- NULL
  
  if (length(classifiers) > 0 ) {
    scores <- compare_classifiers(mydataset, classifiers)
  }  
  
  return (scores)
})  

output$evaluate <- renderTable({
  scores <- classifiersReactive()
})


output$plot <- renderPlot({
  scores <- classifiersReactive()
  scores$Model <- factor(scores$Model, levels = scores$Model[order(scores$"Mean Val. Accuracy")])
  myAccuracy <- as.numeric(scores$"Mean Val. Accuracy")
  p<-ggplot(data=scores, aes(x=Model, y=myAccuracy)) + geom_bar(stat="identity")
  p + coord_flip()
  
})


