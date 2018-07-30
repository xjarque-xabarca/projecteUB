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


## =========================================================================== ##
## CONFIGURING PYTHON FROM R
## =========================================================================== ##     

# Load reticulate package
library(reticulate)

# Configure which version of Python to use
use_python("/opt/anaconda3/bin/python")

# sudo ln -s -f /opt/anaconda3/lib/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1
# ln -s -f /opt/anaconda3/lib/libz.so.1.2.11 /lib/x86_64-linux-gnu/libz.so.1

# Read and evaluate a Python script
source_python("python/DataSetHandler.py")



