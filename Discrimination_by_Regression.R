
# read data into memory
data_set1 <- read.csv("hw02_data_set_images.csv",header = FALSE)
data_set2 <- read.csv("hw02_data_set_labels.csv",header = FALSE)

# get X and y values
X <- as.matrix(data_set1)
y_label <-as.matrix(data_set2)



train_X=rbind(X[1:25,1:320],X[40:64,1:320],X[79:103,1:320],X[118:142,1:320],X[157:181,1:320])
test_X=rbind(X[26:39,1:320],X[65:78,1:320],X[104:117,1:320],X[143:156,1:320],X[182:195,1:320])
train_y<-c(1,125)
test_y<-c(1,70)

for (i in 1:125){
  if (i<=25){
    train_y[i]<-1
  }
  else if(i<=50){
    train_y[i]<-2
  }
  else if(i<=75){
    train_y[i]<-3
  }
  else if(i<=100){
    train_y[i]<-4
  }
  else if(i<=125){
    train_y[i]<-5
  }
}
for (i in 1:70){
  if (i<=14){
    test_y[i]<-1
  }
  else if(i<=28){
    test_y[i]<-2
  }
  else if(i<=42){
    test_y[i]<-3
  }
  else if(i<=56){
    test_y[i]<-4
  }
  else if(i<=70){
    test_y[i]<-5
  }
}




# get number of classes and number of samples
K <- max(train_y)
N <- length(train_y)

# one-of-K-encoding
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, train_y)] <- 1


#define sigmoid function
sigmoid <- function(X, w, w0) {
  return (1 / (1 + exp(-(X %*% w + w0))))
}



# define the gradient functions
gradient_W <- function(X, Y_truth, Y_predicted) {
  return(-t(X)%*%((Y_truth-Y_predicted)*(1-Y_predicted)*(Y_predicted)))
  
}

gradient_w0 <- function(Y_truth, Y_predicted) {
  return (-colSums((Y_truth - Y_predicted)*(1-Y_predicted)*(Y_predicted)))
}

# set learning parameters
eta <- 0.01
epsilon <- 1e-3

# randomly initalize W and w0
set.seed(521)
W <- matrix(runif(ncol(X) * K, min = -0.01, max = 0.01), ncol(X), K)
w0 <- runif(K, min = -0.01, max = 0.01)

# learn W and w0 using gradient descent
iteration <- 1
objective_values <- c()
while (1) {
  Y_predicted <- sigmoid(train_X, W, w0)
  objective_values <- c(objective_values, 0.5*sum((Y_truth - Y_predicted )^2))
  
  W_old <- W
  w0_old <- w0
  
  W <- W - eta * gradient_W(train_X, Y_truth, Y_predicted)
  w0 <- w0 - eta * gradient_w0(Y_truth, Y_predicted)
  
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon) {
    break
  }
  
  iteration <- iteration + 1
}
print(W)
print(w0)


# plot objective function during iterations
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix
y_predicted <- apply(Y_predicted, 1, which.max)
confusion_matrix <- table(y_predicted, train_y)
print(confusion_matrix)




Y_predicted_test <- sigmoid(test_X, W, w0)


y_predicted_test <- apply(Y_predicted_test, 1, which.max)  
confusion_matrix_test <- table(y_predicted_test, test_y)
print(confusion_matrix_test)  

