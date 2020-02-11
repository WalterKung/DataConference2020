#
# This is a Shiny web application. You can run the application by clicking
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(tidyverse)
library(DT)
library(magrittr)


# Define UI for application 
ui <- fluidPage(
    titlePanel("Topic Modeling"),
    sidebarLayout(
        sidebarPanel(
            DT::dataTableOutput("CaS_dataTable"),
            DT::dataTableOutput("CaS_topicTable"),
            textAreaInput("INP_Keywords", NULL, "", height = "220px"),
            tags$head(
                tags$style(
                    "#INP_Keywords{ color:red; font-size:20px; font-family: BentonSans Book; overflow-y:scroll; pre-wrap;}"
                )
            )
        ),
        mainPanel(
            htmlOutput("selected_var"),
            tags$head(
                tags$style(
                    "#selected_var{color:black; font-size:20px; font-family: BentonSans Book; overflow-y:scroll; pre-wrap;}"
                )
            )
        )  
    )
)

# Define server
server <- function(input, output, session) {
    
    # support functions
    wordHighlight <- function(SuspWord,colH = 'yellow') {
        colH = values$color
        paste0(' <span style="background-color:',colH,'">',SuspWord,'</span> ')
    }
    
    myCSV <- reactiveFileReader(1000, session, 'texts.csv', read.csv)
    
    myTopics <- reactiveFileReader(1000, session, 'topics.csv', read.csv)
    
    values <- reactiveValues()
    values$sentence <- ''
    
    observeEvent(
        input$CaS_dataTable_cell_clicked,{
            if (!is.null(input$CaS_dataTable_rows_selected)) { 
                colfunc<-colorRampPalette(c("red","yellow","springgreen","royalblue"))
                
                df <- NULL
                topics <- strsplit(myCSV()[input$CaS_dataTable_rows_selected, 3], ' ')[[1]]
                values$topics <- topics
                coltemplate = colfunc(length(topics))
                i = 0
                last_itopic = -1
                Topics = c()
                for (topic in topics){
                    itopic = as.numeric(gsub("[^0-9]", "", topic)) + 1
                    if (itopic != last_itopic){
                        i = i + 1
                        Topics[i] = sprintf("%03d", itopic - 1)
                        last_itopic = itopic
                    }
                }
                values$topics <- as.data.frame(Topics)
                tmp <- myCSV()[input$CaS_dataTable_rows_selected, 1]
                values$sentence <- tmp
                updateTextAreaInput(session, "INP_Keywords", value = "")
            }
        }
    )
    
    observeEvent(
        input$CaS_topicTable_cell_clicked,{
            if (!is.null(input$CaS_topicTable_rows_selected)) { 
                selectedTopic = values$topics[input$CaS_topicTable_rows_selected,1]
                colfunc<-colorRampPalette(c("springgreen"))
                coltemplate = colfunc(1)
                
                itopic = as.numeric(gsub("[^0-9]", "", selectedTopic)) + 1
                str_e = strsplit(toString(myTopics()[itopic,2]), ' ')[[1]]
                values$keywords = paste(str_e, collapse = '\n')
                
                df <- NULL
                for (s in str_e){
                    df <- rbind(df, data.frame(key=s, topic=itopic, color = coltemplate[1]))                    
                }
                
                tmp <- myCSV()[input$CaS_dataTable_rows_selected, 1]
                for (i in 1:nrow(df)){
                    s = df[i,1]
                    values$color = df[i,3]
                    tmp %<>% str_replace_all(regex(paste("[^a-zA-Z<>]",s,"[^a-zA-Z<>]|^",s,"[^a-zA-Z<>]", sep=''), ignore_case = TRUE), wordHighlight)
                }
                values$sentence <- tmp
                updateTextAreaInput(session, "INP_Keywords", value = trimws(values$keywords))
            }
        }
    )
    
    output$selected_var <- renderText({ 
        values$sentence
    })
    
    ##################################################################    
    # Render Email items  
    ##################################################################    
    output$CaS_dataTable <- DT::renderDataTable({
        DT::datatable(
            myCSV()[,c(2,ncol(myCSV()))], 
            selection='single',
            class = "display pre-wrap compact", # style
            filter = "top",
            options = list(  # options
                scrollX = TRUE, # allow user to scroll wide tables horizontally
                stateSave = FALSE,
                autoWidth = FALSE,
                lengthChange = FALSE,
                filter = TRUE,
                pageLength = 5
            )
        )
        
    })

    ##################################################################    
    # Render Topics items  
    ##################################################################    
    
    output$CaS_topicTable <- DT::renderDataTable({
        DT::datatable(
            values$topics, 
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
    
}

# Run the application 
shinyApp(ui = ui, server = server)




