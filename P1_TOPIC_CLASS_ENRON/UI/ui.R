#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

# Initialization
dir_choices <- list('')
branch_choices <- list('')

# Define UI for application
shinyUI(
  fluidPage(tags$head(tags$style(HTML("pre { white-space: pre-wrap; word-break: keep-all; }"))),
      sidebarLayout(
          sidebarPanel(width = 6,
                    fluidRow(class = "sideRow1", DT::dataTableOutput("CaS_dataTable"), placeholder = TRUE),
                    fluidRow(class = "sideRow2", verbatimTextOutput("CaS_emailBody"), placeholder = TRUE),
                    fluidRow(align = "bottom", class = "sideRow3", 
                              column(10, DT::dataTableOutput("CaS_contactTable")),
                              column(2, actionButton("CaS_confirm_CInfo", ">>"))
                            )
          ),
          mainPanel(width = 6,
            h4("Summary"),
            fluidRow(class = "mainRow1", 
                     column(width = 10, textAreaInput("INP_emailSummary", NULL, "", width = "550px", height = "220px")),
                     column(width = 2, verbatimTextOutput("INP_Dup_n"))
                     ),
            h4("Contact Information"),
            fluidRow(class = "mainRow2", 
                     column(12,style='background-color:#f2f2f2;min-width: 800px;',
                            br(),
                            textInput(inputId = "INP_firstname", label = "First Name :"),
                            textInput(inputId = "INP_lastname", label = "Last Name :"),
                            textInput(inputId = "INP_email", label = "Email Addr :"),
                            br(),
                            actionButton("INP_recommend_actions", "Recommend Actions")
                     )                     
            ),
            h4("Actions"),
            fluidRow(class = "mainRow3"
                     ,column(width = 8, 
                             selectizeInput(inputId = 'INP_assignTo', NULL, choices = dir_choices, selected = NULL, multiple = TRUE),
                             actionButton("INP_moreAssign", "+", style = "font-size:80%"),
                             actionButton("INP_lessAssign", "-", style = "font-size:80%"),
                             textInput(inputId = "INP_Actual", label = "Actual :")
                      )
            ), 
            tags$head(
              tags$style(type="text/css","label{ display: table-cell; text-align: center;vertical-align: middle; font-size: 11px;} .form-group { display: table-row;}") 
            )
          )
        ),
        tags$head(
          tags$style("
                          #CaS_dataTable{color:black; font-size:15px; font-family: BentonSans Book; overflow-y:scroll; max-height:320px} 
                          #CaS_contactTable{color:black; font-size:15px; font-family: BentonSans Book; overflow-y:scroll; max-height:150px} 
                          #CaS_emailBody{color:black; font-size:17px; font-family: BentonSans Book; overflow-y:scroll; max-height:300px; pre-wrap;}
                          #CaS_confirm_CInfo{width:100%; margin-top: 30px;}
                          #INP_emailSummary{color:black; font-size:17px; font-family: BentonSans Book; overflow-y:scroll; max-height:220px; pre-wrap;}
                          #INP_firstname{color:black; font-size:13px; font-family: BentonSans Book; pre-wrap;}
                          #INP_lastname{color:black; font-size:13px; font-family: BentonSans Book; pre-wrap;}
                          #INP_email{color:black; font-size:13px; font-family: BentonSans Book; pre-wrap;}
                          #INP_confirm_contact{color:black; font-size:13px; font-family: BentonSans Book; pre-wrap;}
                          #INP_recommend_actions{color:black; font-size:13px; font-family: BentonSans Book; pre-wrap;}
                          .sideRow1{height:220px}
                          .sideRow2{height:315px}
                          .sideRow3{height:150px}
                          .mainRow1{height:230px}
                          .mainRow2{height:150px}
                          .mainRow3{height:130px, font-size:10px;}

                        "
          )
        )
  )
)
