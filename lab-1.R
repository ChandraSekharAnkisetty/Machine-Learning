# Load necessary libraries
library(caret)
library(dplyr)
library(ggplot2)

# Read data from the Github link
url <- "https://github.com/nasimm48/machine-learning/raw/main/lab-1/data/oulad-students.csv"
data <- read.csv(url)
# Assuming 'data' is your data frame
subset_data <- data[, c("id_student", "date_registration", "module_presentation_length", "studied_credits", "num_of_prev_attempts")]

# Describe the variables
summary(subset_data)

#to viewing the data 
View(data)


# Remove rows with missing values
data <- na.omit(data)

# Convert categorical variables to factors
data$code_module <- as.factor(data$code_module)
data$code_presentation <- as.factor(data$code_presentation)
data$gender <- as.factor(data$gender)
data$region <- as.factor(data$region)
data$highest_education <- as.factor(data$highest_education)
data$imd_band <- as.factor(data$imd_band)
data$age_band <- as.factor(data$age_band)
data$num_of_prev_attempts <- as.factor(data$num_of_prev_attempts)
data$disability <- as.factor(data$disability)
data$final_result <- as.factor(data$final_result)

# Split the data into training and testing sets (80% training, 20% testing)
set.seed(120) # For reproducibility
train_index <- createDataPartition(data$final_result, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Train the classification model (logistic regression)
model <- train(final_result ~ ., data = train_data, method = "glm", family = "binomial")

# Make predictions on the test data
predictions <- predict(model, newdata = test_data)

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data$final_result)

# Convert confusion matrix to a data frame
conf_matrix_df <- as.data.frame(conf_matrix$table)
conf_matrix_df <- cbind(Actual = rownames(conf_matrix_df), conf_matrix_df)
rownames(conf_matrix_df) <- NULL

# Plotting the confusion matrix
ggplot(data = conf_matrix_df, aes(x = Actual, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()

