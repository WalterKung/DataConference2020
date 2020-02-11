  #
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(DT)        
library(qdapRegex)

# initialization
n = c(2, 3, 5, 7, 8, 2, 3, 5, 7, 8) 
s = c("aa", "bb", "cc", "bb", "cccccccccccccccccc", "aa", "bb", "cc", "bb", "cc") 
b = c(TRUE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, TRUE) 
df = data.frame(n, s, b)  
home = ''

# Initialization
dir_choices <- list(
  '[1_1:Company Business(other)]', '[1_2:Personal]', '[1_4:Logistic Arrangements]', 
  '[1_5:Employment arrangements]', '[1_6:Document editing]', '[1_7:Empty]',
  '[3_1:Legal]', '[3_10:Regulations]', '[3_2:Projects]', '[3_3:company image]', '[3_5:Political]', 
  '[3_6:California Energy]', '[3_7:Internal op/policy]' 
)


# loading python library
#-----------------------------------------

#-----------------------------------------

assignto_getProbLab <- function(probtxt){
  label = unlist(dir_choices)
  prob = scan(text = gsub("[^0-9A-Za-z////.' ]"," " , probtxt ,ignore.case = TRUE))
  return (data.frame(label, prob))
}

# Define server logic 
shinyServer(function(input, output, session) {
  # initiate values
  # Create the object with no values
  values <- reactiveValues()
  # Assign values to text and probThershold to pass value from module to module
  values$text <- ""
  values$probThershold <- 0.25
  # load reactive CSV file
  myCSV <- reactiveFileReader(1000, session, paste(home, "emails_sum.csv", sep = ""), read.csv)
  myContact <- reactiveFileReader(1000, session, paste(home, "contacts.csv", sep = ""), read.csv)
  values$contactEmailAddr <- ""
  values$assignto <- ""
  
  ##################################################################    
  # Event Listener
  ##################################################################    
  observeEvent(
      #-------------------------------------------------------------    
      # Confirm contact
      #-------------------------------------------------------------    
      input$CaS_confirm_CInfo, {
          if(length(input$CaS_contactTable_rows_selected) > 0) {
              buf = values$contactdf[input$CaS_contactTable_rows_selected,"FNAME"]
              if (buf =="%%UNK%%"){buf = ""}
              updateTextInput(session, "INP_firstname", label = NULL, value = buf)
            
              buf = values$contactdf[input$CaS_contactTable_rows_selected,"LNAME"]
              if (buf =="%%UNK%%"){buf = ""}
              updateTextInput(session, "INP_lastname", label = NULL, value = buf)
            
              buf = values$contactdf[input$CaS_contactTable_rows_selected,"EMAIL"]
              if (buf =="%%UNK%%"){buf = ""}
              updateTextInput(session, "INP_email", label = NULL, value = buf)
              values$contactEmailAddr <- buf
            
          }
     }
  )

  observeEvent(
    input$INP_recommend_actions, {
        if (length(values$text) > 0) {
          text <- as.character(values$text)
          rst_prob <- assignto_getProbLab(values$score)
          selected_ <- rst_prob$label[rst_prob$prob > values$probThershold]
        } else{
            selected_ <- NULL
        }
        updateSelectizeInput(session, "INP_assignTo", label = "Assign To", choices = dir_choices, selected = selected_)
    }
  )
  
  observeEvent(
    input$INP_moreAssign, {
      if (values$probThershold > 0.16)
      {
        values$probThershold <- values$probThershold - 0.07
        ### Fix this it is dup
        if (length(values$text) > 0) {
          text <- as.character(values$text)
          rst_prob <- assignto_getProbLab(values$score)
          selected_ <- rst_prob$label[rst_prob$prob > values$probThershold]
        } else{
            selected_ <- NULL
        }
        updateSelectizeInput(session, "INP_assignTo", label = "Assign To", choices = dir_choices, selected = selected_)
      }
    }, ignoreInit = TRUE)
  
  observeEvent(
    input$INP_lessAssign, {
      if (values$probThershold < 0.8)
      {
        values$probThershold <- values$probThershold + 0.07
        ### Fix this it is dup
        if (length(values$text) > 0) {
            text <- as.character(values$text)
            rst_prob <- assignto_getProbLab(values$score)
            selected_ <- rst_prob$label[rst_prob$prob > values$probThershold]
        } else{
            selected_ <- NULL
        }
        updateSelectizeInput(session, "INP_assignTo", label = "Assign To", choices = dir_choices, selected = selected_)
      }
    }, ignoreInit = TRUE)

  
  
  ##################################################################    
  # Render Email items  
  ##################################################################    
  output$CaS_dataTable <- DT::renderDataTable({
    DT::datatable(
      myCSV()[,c(12,4,6)], 
      selection='single',
      class = "display pre-wrap compact", # style
      options = list(  # options
        scrollX = TRUE, # allow user to scroll wide tables horizontally
        stateSave = FALSE,
        autoWidth = FALSE,
        lengthChange = FALSE,
        filter = FALSE,
        pageLength = 2
      )
    )
  })
  
  ##################################################################    
  # Fill in Email body
  ##################################################################    
  output$CaS_emailBody <- renderText({
    st <- as.character(factor(myCSV()[input$CaS_dataTable_rows_selected,"SUBJECT"]))
    e <- as.character(factor(myCSV()[input$CaS_dataTable_rows_selected,"text"]))
    nm <- as.character(factor(myCSV()[input$CaS_dataTable_rows_selected,"FNAME"]))
    nm <- paste(nm, as.character(factor(myCSV()[input$CaS_dataTable_rows_selected,"LNAME"])))
    values$EmailAddr <- as.character(factor(myCSV()[input$CaS_dataTable_rows_selected,"EMAIL"]))
    values$CCAddr <-  ""
    values$master_rootid <- as.character(factor(myCSV()[input$CaS_dataTable_rows_selected,"master_rootid"]))
    values$score <- as.character(factor(myCSV()[input$CaS_dataTable_rows_selected,"score"]))
    values$assignto <- as.character(factor(myCSV()[input$CaS_dataTable_rows_selected,"ASGNTO"]))
        
    if (is.null(e)) {e <- " "}
    if (length(e)<1) {e <- " "}
    if (nchar(e)<1) {e <- " "}
    values$text <- e
    buf <- tryCatch(
          {as.character(as.character(factor(myCSV()[input$CaS_dataTable_rows_selected,"summary"])))}, 
          error=function(cond){return (values$text)},
          warning=function(cond){return (values$text)}
          )
    if (nchar(buf) < 2){buf = values$text}
    updateTextAreaInput(session, "INP_emailSummary", value = trimws(buf))
    values$probThershold <- 0.5

    updateTextInput(session, "INP_firstname", label = NULL, value = " ")
    updateTextInput(session, "INP_lastname", label = NULL, value = " ")
    updateTextInput(session, "INP_email", label = NULL, value = " " )
    
    updateTextInput(session, "INP_Actual", label = NULL, value = values$assignto )
    updateSelectizeInput(session, "INP_assignTo", label = "Assign To", choices = dir_choices, selected = NULL)
    values$contactEmailAddr <- ""
    trimws(values$text)
  })

  ##################################################################    
  # Fill in the potential contact
  ##################################################################    
  output$CaS_contactTable <- DT::renderDataTable({
    s <- paste(values$text, " ", values$EmailAddr, " ", values$CCAddr)
    values$contactdf <- myContact()[myContact()[,"EMAIL"] %in% unlist(lapply(ex_email(s), toupper)), ]
    DT::datatable(
      values$contactdf, 
      selection='single',
      class = "display pre-wrap compact", # style
      options = list(  # options
        scrollX = TRUE, # allow user to scroll wide tables horizontally
        stateSave = FALSE,
        autoWidth = FALSE,
        lengthChange = FALSE,
        filter = FALSE,
        pageLength = 3
      )
    )
  })

  #-----------------------------------------------------------------------------
  #  Output Email details
  #-----------------------------------------------------------------------------


})
