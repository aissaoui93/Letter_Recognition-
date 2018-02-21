Letter Recognition
================
Ahmed Issaoui
2/18/2018

##### One of the most widespread applications of machine learning is to do optical character/letter recognition, which is used in applications from physical sorting of mail at post offices, to reading license plates at toll booths, to processing check deposits at ATMs. Optical character recognition by now involves very well-tuned and sophisticated models.

##### In this problem, we will build a simple model that uses attributes of images of four letters in the Roman alphabet - A, B, P, and R - to predict which letter a particular image corresponds to. In this problem, we have four possible classifications of each data observation, namely whether the observation is the letter A, B, P, or R. Such problems are called multi-class classification problems.

##### The data set contains 3116 observations, each of which corresponds to a certain image of one of the four letters A, B, P and R. The images came from 20 different fonts, which were then randomly distorted to produce the final images; each such distorted image is represented as a collection of pixels, and each pixel is either "on" or "off. For each such distorted image, we have available certain attributes of the image in terms of these pixels, as well as which of the four letters the image is. These features are described below:

-   xbox : The horizontal position of where the smallest box enclosing the letter shape begins.

-   ybox : The vertical position of where the smallest box enclosing the letter shape begins.

-   width : The width of this smallest box.

-   height : The height of this smallest box.

-   onpix : The total number of " pixels in the character image.

-   xbar : The mean horizontal position of all of the "on" pixels.

-   ybar : The mean vertical position of all of the "on" pixels.

-   x2bar : The mean squared horizontal position of all of the "on" pixels in the image.

-   y2bar : The mean squared vertical position of all of the "on" pixels in the image.

-   xybar : The mean of the product of the horizontal and vertical position of all of the "on" pixels in the image.

-   x2ybar : The mean of the product of the squared horizontal position and the vertical position of all of the "on" pixels.

-   xy2bar : The mean of the product of the horizontal position and the squared vertical position of all of the "on" pixels.

-   xedge : The mean number of edges (the number of times an "off"" pixel is followed by an "on" pixel, or the image boundary is hit) as the image is scanned from left to right, along the whole vertical length of the image.

-   xedgeycor : The mean of the product of the number of horizontal edges at each vertical position and the vertical position.

-   yedge : The mean number of edges as the images is scanned from top to bottom, along the whole horizontal length of the image.

-   yedgexcor : The mean of the product of the number of vertical edges at each horizontal position and the horizontal position.

Let's look at the first rows of the dataframe "letters"

``` r
head(letters)
```

    ##   letter xbox ybox width height onpix xbar ybar x2bar y2bar xybar x2ybar
    ## 1      B    4    2     5      4     4    8    7     6     6     7      6
    ## 2      A    1    0     3      2     1    8    2     2     2     8      2
    ## 3      R    5    9     5      7     5    5   11     7     3     7      3
    ## 4      B    5    9     7      7    10    9    8     4     4     6      8
    ## 5      P    3    6     4      4     2    5   14     8     2    11      6
    ## 6      R    8   10     8      6     6    7    7     3     5     8      4
    ##   xy2bar xedge xedgeycor yedge yedgexcor
    ## 1      6     2         8     7        10
    ## 2      8     1         6     2         7
    ## 3      9     2         7     5        11
    ## 4      6     6        11     8         7
    ## 5      3     0        10     4         8
    ## 6      8     6         6     7         7

What about the types of each column?

``` r
str(letters)
```

    ## 'data.frame':    3116 obs. of  17 variables:
    ##  $ letter   : Factor w/ 4 levels "A","B","P","R": 2 1 4 2 3 4 4 1 3 3 ...
    ##  $ xbox     : int  4 1 5 5 3 8 2 3 8 6 ...
    ##  $ ybox     : int  2 0 9 9 6 10 6 7 14 10 ...
    ##  $ width    : int  5 3 5 7 4 8 4 5 7 7 ...
    ##  $ height   : int  4 2 7 7 4 6 4 5 8 8 ...
    ##  $ onpix    : int  4 1 5 10 2 6 3 3 4 7 ...
    ##  $ xbar     : int  8 8 5 9 5 7 6 12 5 8 ...
    ##  $ ybar     : int  7 2 11 8 14 7 7 2 10 5 ...
    ##  $ x2bar    : int  6 2 7 4 8 3 5 3 6 7 ...
    ##  $ y2bar    : int  6 2 3 4 2 5 5 2 3 6 ...
    ##  $ xybar    : int  7 8 7 6 11 8 5 10 12 7 ...
    ##  $ x2ybar   : int  6 2 3 8 6 4 6 2 5 6 ...
    ##  $ xy2bar   : int  6 8 9 6 3 8 7 9 4 6 ...
    ##  $ xedge    : int  2 1 2 6 0 6 3 2 4 3 ...
    ##  $ xedgeycor: int  8 6 7 11 10 6 8 6 10 9 ...
    ##  $ yedge    : int  7 2 5 8 4 7 5 3 4 8 ...
    ##  $ yedgexcor: int  10 7 11 7 8 7 9 8 8 9 ...

Let's now look at the main statistics about its columns (max, min, average,...) using pastecs library

``` r
stat.desc(letters[,-1])
```

    ##                      xbox         ybox        width       height
    ## nbr.val      3.116000e+03 3.116000e+03 3.116000e+03 3.116000e+03
    ## nbr.null     9.000000e+00 1.060000e+02 0.000000e+00 6.000000e+01
    ## nbr.na       0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
    ## min          0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
    ## max          1.300000e+01 1.500000e+01 1.100000e+01 1.200000e+01
    ## range        1.300000e+01 1.500000e+01 1.000000e+01 1.200000e+01
    ## sum          1.216500e+04 2.195900e+04 1.614400e+04 1.643800e+04
    ## median       4.000000e+00 7.000000e+00 5.000000e+00 6.000000e+00
    ## mean         3.904044e+00 7.047176e+00 5.181001e+00 5.275353e+00
    ## SE.mean      3.334181e-02 6.037081e-02 3.308685e-02 4.050372e-02
    ## CI.mean.0.95 6.537415e-02 1.183706e-01 6.487424e-02 7.941669e-02
    ## var          3.463984e+00 1.135668e+01 3.411209e+00 5.111957e+00
    ## std.dev      1.861178e+00 3.369968e+00 1.846946e+00 2.260964e+00
    ## coef.var     4.767308e-01 4.782012e-01 3.564843e-01 4.285900e-01
    ##                     onpix         xbar         ybar        x2bar
    ## nbr.val      3.116000e+03 3.116000e+03 3.116000e+03 3.116000e+03
    ## nbr.null     8.200000e+01 0.000000e+00 9.000000e+00 3.000000e+00
    ## nbr.na       0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
    ## min          0.000000e+00 3.000000e+00 0.000000e+00 0.000000e+00
    ## max          1.200000e+01 1.400000e+01 1.500000e+01 1.100000e+01
    ## range        1.200000e+01 1.100000e+01 1.500000e+01 1.100000e+01
    ## sum          1.206000e+04 2.326800e+04 2.241700e+04 1.466900e+04
    ## median       4.000000e+00 7.000000e+00 7.000000e+00 4.000000e+00
    ## mean         3.870347e+00 7.467266e+00 7.194159e+00 4.707638e+00
    ## SE.mean      3.914012e-02 3.388942e-02 5.067102e-02 4.045652e-02
    ## CI.mean.0.95 7.674305e-02 6.644786e-02 9.935198e-02 7.932414e-02
    ## var          4.773554e+00 3.578703e+00 8.000492e+00 5.100051e+00
    ## std.dev      2.184846e+00 1.891746e+00 2.828514e+00 2.258329e+00
    ## coef.var     5.645092e-01 2.533385e-01 3.931681e-01 4.797160e-01
    ##                     y2bar        xybar       x2ybar       xy2bar
    ## nbr.val      3.116000e+03 3.116000e+03 3.116000e+03 3.116000e+03
    ## nbr.null     1.140000e+02 0.000000e+00 5.500000e+01 1.000000e+00
    ## nbr.na       0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
    ## min          0.000000e+00 3.000000e+00 0.000000e+00 0.000000e+00
    ## max          8.000000e+00 1.400000e+01 1.000000e+01 1.400000e+01
    ## range        8.000000e+00 1.100000e+01 1.000000e+01 1.400000e+01
    ## sum          1.217700e+04 2.647800e+04 1.407800e+04 2.092400e+04
    ## median       4.000000e+00 8.000000e+00 5.000000e+00 7.000000e+00
    ## mean         3.907895e+00 8.497433e+00 4.517972e+00 6.715019e+00
    ## SE.mean      3.349942e-02 3.671068e-02 3.562975e-02 3.638362e-02
    ## CI.mean.0.95 6.568319e-02 7.197957e-02 6.986017e-02 7.133829e-02
    ## var          3.496811e+00 4.199351e+00 3.955696e+00 4.124859e+00
    ## std.dev      1.869976e+00 2.049232e+00 1.988893e+00 2.030975e+00
    ## coef.var     4.785124e-01 2.411589e-01 4.402182e-01 3.024526e-01
    ##                     xedge    xedgeycor        yedge    yedgexcor
    ## nbr.val      3.116000e+03 3.116000e+03 3.116000e+03 3.116000e+03
    ## nbr.null     1.110000e+02 0.000000e+00 1.000000e+01 0.000000e+00
    ## nbr.na       0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
    ## min          0.000000e+00 1.000000e+00 0.000000e+00 1.000000e+00
    ## max          1.000000e+01 1.300000e+01 1.200000e+01 1.300000e+01
    ## range        1.000000e+01 1.200000e+01 1.200000e+01 1.200000e+01
    ## sum          9.078000e+03 2.417000e+04 1.434000e+04 2.622200e+04
    ## median       2.000000e+00 8.000000e+00 4.000000e+00 8.000000e+00
    ## mean         2.913350e+00 7.756739e+00 4.602054e+00 8.415276e+00
    ## SE.mean      3.239885e-02 3.002043e-02 3.658074e-02 3.021716e-02
    ## CI.mean.0.95 6.352526e-02 5.886183e-02 7.172481e-02 5.924756e-02
    ## var          3.270820e+00 2.808221e+00 4.169678e+00 2.845147e+00
    ## std.dev      1.808541e+00 1.675775e+00 2.041979e+00 1.686756e+00
    ## coef.var     6.207770e-01 2.160411e-01 4.437103e-01 2.004398e-01

Do we have any missing values in our dataframe?

``` r
colSums(is.na(letters))
```

    ##    letter      xbox      ybox     width    height     onpix      xbar 
    ##         0         0         0         0         0         0         0 
    ##      ybar     x2bar     y2bar     xybar    x2ybar    xy2bar     xedge 
    ##         0         0         0         0         0         0         0 
    ## xedgeycor     yedge yedgexcor 
    ##         0         0         0

Let's now using histograms to describe the frequency of the values of some columns of the dataframe. Here we choose to look at some columns: "xbox", "x2bar", and "xedgeycor"

``` r
hist(letters$xbox, xlab = "xbox", breaks = 15, main = "xbox distribution")
```

![](Letter_Recognition_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-1.png)

``` r
hist(letters$x2bar, xlab = "x2bar", breaks = 15, main = "x2bar distribution")
```

![](Letter_Recognition_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-2.png)

``` r
hist(letters$xedgeycor, xlab = "xedgeycor", breaks = 15, main = "xedgeycor distribution")
```

![](Letter_Recognition_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-3.png)

Let's see if "xedgeycor = The mean of the product of the number of horizontal edges at each vertical position and the vertical position" is diffrent from one lette rto the other between the four letter A, B, P and R.

``` r
plot(letters$xedgeycor, letters$letter , main = "scatter plot 'letter = f(xedgeycor)'",xlab = "xedgeycor", ylab = "letter A, B, P or R")
```

![](Letter_Recognition_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-6-1.png)

Let's compare the density of the same three columns selected earlier ("xbox", "x2bar", and "xedgeycor") between the four letter A, B, P and R.

``` r
sm.density.compare(letters$xbar, letters$letter)
```

![](Letter_Recognition_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-1.png)

``` r
sm.density.compare(letters$x2bar, letters$letter)
```

![](Letter_Recognition_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-2.png)

``` r
sm.density.compare(letters$xedgeycor, letters$letter)
```

![](Letter_Recognition_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-7-3.png)

-   Let's create a new variable called isB, which takes value "Yes" if the letter is B, and "No" otherwise

``` r
letters$isB <- ifelse(letters$letter == "B" , "Yes", "No")

set.seed(623)

# randomly split the dataset into a training set and a test set
train_ids = sample(nrow(letters), 0.65*nrow(letters))
letters_train = letters[train_ids,]
letters_test = letters[-train_ids,]
```

What is the accuracy of this baseline method on the train set?

    ## 
    ##   No  Yes 
    ## 1523  502

    ## Accuracy of the baseline model on the training set = 0.8477157

What is the accuracy of this baseline method on the test set?

``` r
# Accuracy of baseline on testing:
table(letters_test$isB)
```

    ## 
    ##  No Yes 
    ## 827 264

``` r
cat("Accuracy of the baseline model on testing set=",930/(930+167))
```

    ## Accuracy of the baseline model on testing set= 0.8477666

``` r
# since the logistic regression requires the dependent variable y values to be 0 <= y <= 1
letters_train$isB <- ifelse(letters_train$letter == "B" , 1, 0)

# Construct a logistic regression model to predict whether or not the letter is a B

log_mod <- glm(isB ~ ., data = letters_train[,-1], family = "binomial")
summary(log_mod)
```

    ## 
    ## Call:
    ## glm(formula = isB ~ ., family = "binomial", data = letters_train[, 
    ##     -1])
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -3.1858  -0.1771  -0.0242   0.0000   3.3735  
    ## 
    ## Coefficients:
    ##               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -10.837601   2.357573  -4.597 4.29e-06 ***
    ## xbox         -0.020443   0.114608  -0.178 0.858429    
    ## ybox          0.041565   0.084701   0.491 0.623619    
    ## width        -1.141151   0.149116  -7.653 1.97e-14 ***
    ## height       -0.677233   0.134686  -5.028 4.95e-07 ***
    ## onpix         0.952260   0.129698   7.342 2.10e-13 ***
    ## xbar          0.465464   0.126289   3.686 0.000228 ***
    ## ybar         -0.682545   0.117880  -5.790 7.03e-09 ***
    ## x2bar        -0.316760   0.091239  -3.472 0.000517 ***
    ## y2bar         1.348638   0.121468  11.103  < 2e-16 ***
    ## xybar         0.245001   0.088372   2.772 0.005565 ** 
    ## x2ybar        0.457329   0.123073   3.716 0.000202 ***
    ## xy2bar       -0.594978   0.104513  -5.693 1.25e-08 ***
    ## xedge        -0.247664   0.088767  -2.790 0.005270 ** 
    ## xedgeycor     0.007102   0.106466   0.067 0.946818    
    ## yedge         1.647975   0.124537  13.233  < 2e-16 ***
    ## yedgexcor     0.308206   0.068692   4.487 7.23e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 2268.07  on 2024  degrees of freedom
    ## Residual deviance:  674.18  on 2008  degrees of freedom
    ## AIC: 708.18
    ## 
    ## Number of Fisher Scoring iterations: 8

Let's see the results of the logitic regression predictions:

``` r
# Predictions on the test set
predTest_log = predict(log_mod, newdata = letters_test[,-1], type = "response")


# Lets create a confusion matrix with a threshhold of probability = 0.5
table(letters_test$isB, predTest_log > 0.5)
```

    ##      
    ##       FALSE TRUE
    ##   No    799   28
    ##   Yes    35  229

So what is the accuracy of the logistic regression model?

    ## The accuracy of the logistic regression model is =  0.9422548

What is the AUC of your logistic regression model?

``` r
# What is the AUC of your logistic regression model?
rocr_log_pred <- prediction(predTest_log, letters_test$isB)
cat("The AUC of your logistic regression model = ",
    as.numeric(performance(rocr_log_pred, "auc")@y.values))
```

    ## The AUC of your logistic regression model =  0.9821141

let's train a CART model:

``` r
# Here we cross validate to find the best value of cp between 0 and 1, that minimizes the accuracy
library(e1071)
train_cart <- train(as.factor(isB) ~.,
                  data = letters_train[,-1],
                  method="rpart",
                  tuneGrid = data.frame(cp=seq(0,0.1,0.005)),
                  minbucket=5,
                  trControl=trainControl(method = "cv", number = 5),
                  metric="Accuracy")
```

Here is the result of the cross validation:

``` r
train_cart
```

    ## CART 
    ## 
    ## 2025 samples
    ##   16 predictor
    ##    2 classes: '0', '1' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1621, 1619, 1620, 1619, 1621 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp     Accuracy   Kappa    
    ##   0.000  0.9338244  0.8242641
    ##   0.005  0.9343255  0.8242158
    ##   0.010  0.9274168  0.8033536
    ##   0.015  0.9195290  0.7783905
    ##   0.020  0.9130933  0.7569003
    ##   0.025  0.9101425  0.7485325
    ##   0.030  0.9061870  0.7334042
    ##   0.035  0.8938132  0.6927318
    ##   0.040  0.8923329  0.6860531
    ##   0.045  0.8873947  0.6680669
    ##   0.050  0.8834538  0.6492184
    ##   0.055  0.8745600  0.6335865
    ##   0.060  0.8696095  0.6258472
    ##   0.065  0.8543019  0.5885420
    ##   0.070  0.8543019  0.5885420
    ##   0.075  0.8543019  0.5885420
    ##   0.080  0.8543019  0.5885420
    ##   0.085  0.8483760  0.5758386
    ##   0.090  0.8483760  0.5758386
    ##   0.095  0.8483760  0.5758386
    ##   0.100  0.8483760  0.5758386
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was cp = 0.005.

And here are the results of the CART model prediction with the best parameter cp

``` r
cart_mod <- rpart(as.factor(isB) ~.,
             data = letters_train[,-1],
             method="class",
             minbucket=5,
             cp = 0)
# Predictions on the test set
predTest_cart = predict(cart_mod, newdata = letters_test[,-1], type = "class")
table(letters_test$isB, predTest_cart)
```

    ##      predTest_cart
    ##         0   1
    ##   No  797  30
    ##   Yes  37 227

    ## The accuracy of the CART model is =  0.9385885

Now let's try random forests:

``` r
rf_mod = randomForest(as.factor(isB) ~ .,
                      data=letters_train[,-1],
                      importance = TRUE)
```

Here is the confusion matrix on the training data:

``` r
rf_mod$confusion
```

    ##      0   1 class.error
    ## 0 1504  19  0.01247538
    ## 1   25 477  0.04980080

And here is the importance of each feature:

``` r
importance(rf_mod)
```

    ##                  0        1 MeanDecreaseAccuracy MeanDecreaseGini
    ## xbox      15.18505 12.46818             20.33793         17.12060
    ## ybox      14.34118 14.09372             20.31132         17.25482
    ## width     19.43010 12.63617             22.00485         18.12098
    ## height    16.79260 11.69268             19.59678         16.41580
    ## onpix     12.85704 15.50885             19.22292         17.50407
    ## xbar      13.54576 22.47233             23.39131         21.17633
    ## ybar      20.14079 29.44365             29.28807         54.92325
    ## x2bar     21.45820 23.28772             28.52082         28.33984
    ## y2bar     24.03012 45.70183             46.03706        106.92346
    ## xybar     25.55421 30.04449             33.38989         40.37948
    ## x2ybar    19.12138 23.32977             25.32186         43.24705
    ## xy2bar    24.98172 33.17754             36.28960         63.51788
    ## xedge     25.62423 26.67356             34.70327         43.38500
    ## xedgeycor 28.35553 34.92239             38.72877         85.47576
    ## yedge     30.53983 49.40187             45.44275        135.77019
    ## yedgexcor 21.84212 20.83502             25.38190         46.01110

Now, what is the accuracy of the random forest model on the test set?

    ##      predTest_rf
    ##         0   1
    ##   No  816  11
    ##   Yes  24 240

    ## The accuracy of the Random Forest model is =  0.9679193

    ## - The accuracy of the Random Forest model is =  0.9679193

    ## - The accuracy of the CART model is =  0.9385885

    ## - The random forest model has a higher accuracy

    ##  -->   in this application, accuracy is more important than interpretability,
    ##     because weare not really interested in understanding why is it B or not.
    ##     All what we want to know is whether it is a B or not. As a consequence,
    ##     although random forest are les interpretable than CART model,we still
    ##     think that they are better since they have higher accuracy

let's look at the number of each letter among all absoervations in the test set

    ## 
    ##   A   B   P   R 
    ## 298 264 258 271

    ## the most frequent class is A. As a consequence a baseline model would predict 'A' for all observations

What is the accuracy of the baseline model?

    ## The accuracy of the baseline model is =  0.2731439

Now let's try LDA:

    ## Loading required package: MASS

    ## Warning: package 'MASS' was built under R version 3.4.2

    ## 
    ## Attaching package: 'MASS'

    ## The following object is masked from 'package:sm':
    ## 
    ##     muscle

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     select

What is the accuracy of the model?

    ##    
    ##     FALSE TRUE
    ##   A   298    0
    ##   B    30  234
    ##   P   250    8
    ##   R   243   28

    ## The accuracy of the LDA model on the test set is =  0.2474794

let's train a CART model:

``` r
# Here we cross validate to find the best value of cp between 0 and 1, that minimizes the accuracy
train_cart <- train(as.factor(letter) ~.,
                    data = letters_train[,-18],
                    method="rpart",
                    tuneGrid = data.frame(cp=seq(0,0.1,0.005)),
                    minbucket=5,
                    trControl=trainControl(method = "cv", number = 5),
                    metric="Accuracy")
```

Here is the result of the cross validation:

``` r
train_cart
```

    ## CART 
    ## 
    ## 2025 samples
    ##   16 predictor
    ##    4 classes: 'A', 'B', 'P', 'R' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1620, 1620, 1619, 1621, 1620 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp     Accuracy   Kappa    
    ##   0.000  0.8859415  0.8478332
    ##   0.005  0.8770550  0.8360056
    ##   0.010  0.8617537  0.8155669
    ##   0.015  0.8385401  0.7844963
    ##   0.020  0.8178017  0.7566204
    ##   0.025  0.8182955  0.7572124
    ##   0.030  0.8182955  0.7572124
    ##   0.035  0.8182955  0.7572124
    ##   0.040  0.8182955  0.7572124
    ##   0.045  0.8182955  0.7572124
    ##   0.050  0.8182955  0.7572124
    ##   0.055  0.8182955  0.7572124
    ##   0.060  0.8182955  0.7572124
    ##   0.065  0.8182955  0.7572124
    ##   0.070  0.8182955  0.7572124
    ##   0.075  0.8182955  0.7572124
    ##   0.080  0.8182955  0.7572124
    ##   0.085  0.8182955  0.7572124
    ##   0.090  0.8182955  0.7572124
    ##   0.095  0.8182955  0.7572124
    ##   0.100  0.8182955  0.7572124
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was cp = 0.

And here are the results of the CART model prediction with the best parameter cp

``` r
cart_mod <- rpart(as.factor(letter) ~.,
                  data = letters_train[,-18],
                  method="class",
                  minbucket=5,
                  cp = 0)
# Predictions on the test set
predTest_cart = predict(cart_mod, newdata = letters_test[,-18], type = "class")
table(letters_test$letter, predTest_cart)
```

    ##    predTest_cart
    ##       A   B   P   R
    ##   A 280   6   3   9
    ##   B   1 229   9  25
    ##   P   4   7 245   2
    ##   R   4  28   8 231

    ## The accuracy of the CART model is =  0.9028414

Again, let's try the random forest model:

``` r
rf_mod = randomForest(as.factor(letter) ~ .,
                      data=letters_train[,-18],
                      importance = TRUE)
```

Here is the confusion matrix on the training data:

``` r
rf_mod$confusion
```

    ##     A   B   P   R class.error
    ## A 487   1   1   2  0.00814664
    ## B   1 486   1  14  0.03187251
    ## P   0   5 537   3  0.01467890
    ## R   0  22   1 464  0.04722793

And here is the importance of each feature:

``` r
importance(rf_mod)
```

    ##                   A        B         P         R MeanDecreaseAccuracy
    ## xbox       6.678509 12.28032  9.628279 12.493744             19.58773
    ## ybox      10.267616 16.97010 12.894073 10.822688             26.64003
    ## width      6.524067 10.44231  6.780936 10.957515             16.08718
    ## height     7.850860 12.04750 16.071081  9.894243             24.01424
    ## onpix      8.556558 15.72968  9.850744  9.533117             20.08650
    ## xbar      10.873944 20.60678 11.625490 17.436587             26.66216
    ## ybar      33.521248 32.19236 25.177411 42.349672             44.75120
    ## x2bar     19.128359 23.83414 18.176433 27.042187             32.60362
    ## y2bar     25.784799 41.47282 23.558892 25.118603             45.02351
    ## xybar     17.129406 31.91111 22.512440 31.216188             40.32756
    ## x2ybar    22.488450 25.20944 17.894400 34.908176             35.66425
    ## xy2bar    18.896438 36.02567 30.463931 33.360162             40.98167
    ## xedge     13.016862 28.75957 16.186631 29.074343             37.75981
    ## xedgeycor 19.681029 45.07444 38.963744 58.084259             52.26210
    ## yedge     17.486974 44.13670 21.646041 33.631078             47.52557
    ## yedgexcor 23.855870 20.49881 14.323472 27.883497             34.04203
    ##           MeanDecreaseGini
    ## xbox              19.54896
    ## ybox              21.19147
    ## width             16.10754
    ## height            18.03376
    ## onpix             20.06519
    ## xbar              32.70042
    ## ybar             233.24621
    ## x2bar             61.41948
    ## y2bar            119.94355
    ## xybar             80.34051
    ## x2ybar           148.68465
    ## xy2bar           196.22641
    ## xedge             66.85129
    ## xedgeycor        286.04495
    ## yedge            122.65048
    ## yedgexcor         73.56635

Now, what is the accuracy of the random forest model on the test set?

    ##    predTest_rf
    ##       A   B   P   R
    ##   A 295   0   2   1
    ##   B   0 251   3  10
    ##   P   0   5 252   1
    ##   R   0  10   0 261

    ## The accuracy of the Random Forest model with a default value for mtry parameter is =  0.9706691

Now, let's use cross-validation to select the mtry value for the Random Forests method

Here we use the mtry that maximizes the accuracy: since it is a classification, we can not use RMSE as a metric to minimize for the cross validation.

``` r
train_rf <- train(as.factor(letter) ~.,
                    data = letters_train[,-18],
                    method="rf",
                    tuneGrid = data.frame(mtry=seq(1,10,1)),
                    minbucket=5,
                    trControl=trainControl(method = "cv", number = 5),
                    metric="Accuracy")


train_rf
```

    ## Random Forest 
    ## 
    ## 2025 samples
    ##   16 predictor
    ##    4 classes: 'A', 'B', 'P', 'R' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1619, 1619, 1621, 1620, 1621 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    1    0.9698618  0.9597940
    ##    2    0.9738161  0.9650663
    ##    3    0.9738112  0.9650587
    ##    4    0.9728235  0.9637418
    ##    5    0.9703507  0.9604442
    ##    6    0.9713396  0.9617602
    ##    7    0.9663964  0.9551693
    ##    8    0.9659026  0.9545085
    ##    9    0.9624482  0.9498981
    ##   10    0.9589914  0.9452860
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 2.

Now let's train the random forest with the best mtry parmeter:

``` r
rf_mod = randomForest(as.factor(letter) ~ .,
                      data=letters_train[,-18],
                      importance = TRUE,
                      mtry = 2)
```

Now, what is the accuracy of the Random Forest model with the best paramter mtry?

    ##    predTest_rf
    ##       A   B   P   R
    ##   A 295   0   1   2
    ##   B   0 253   3   8
    ##   P   0   4 252   2
    ##   R   0  12   0 259

    ## The accuracy of the Random Forest model with the best parameter mtry is =  0.9743355

The best value of mtry is 2. But what is the default valut of mtry used by random forest model in question c.iv?

    ## the default value of mtry is =  4

    ## [1] " . The best parmater mtry is then smaller than the default value"

Now let's try the boosting model:

``` r
# training the model
boost_mod <- gbm(letter~.,
                 data = letters_train[,-18],
                 distribution = "multinomial",
                 n.trees = 22400,
                 interaction.depth = 10)
```

What about its accuracy?

``` r
## work around bug in gbm 2.1.1
predict.gbm <- function (object, newdata, n.trees, type = "link", single.tree = FALSE, ...) {
  if (missing(n.trees)) {
    if (object$train.fraction < 1) {
      n.trees <- gbm.perf(object, method = "test", plot.it = FALSE)
    }
    else if (!is.null(object$cv.error)) {
      n.trees <- gbm.perf(object, method = "cv", plot.it = FALSE)
    }
    else {
      n.trees <- length(object$train.error)
    }
    cat(paste("Using", n.trees, "trees...\n"))
    gbm::predict.gbm(object, newdata, n.trees, type, single.tree, ...)
  }
}

# matrix of the probability estimate (as predicted by the gradient boosting model) for each of the 4 classes
blah = predict(boost_mod, newdata = letters_test[,-18], type = "response")
```

    ## Using 22400 trees...

``` r
# convert this matrix to a vector of class predictions
predTest_boost = apply(blah, 1, which.max)
predTest_boost = factor(predTest_boost, levels = c(1,2,3,4), labels = c("A", "B", "P", "R"))

# What is the accuracy of the Random Forest model?
table(letters_test$letter, predTest_boost)
```

    ##    predTest_boost
    ##       A   B   P   R
    ##   A 295   0   1   2
    ##   B   1 249   2  12
    ##   P   0   2 254   2
    ##   R   0  10   0 261

``` r
cat("The accuracy of the boosting model is = ", (295+249+254+261)/nrow(letters_test))
```

    ## The accuracy of the boosting model is =  0.9706691

So now, let's compare the accuracies of our LDA, CART, Random Forest, and boosting models for this problem.

    ## The accuracy of the baseline model on the test set is =  0.2731439

    ## The accuracy of the LDA model on the test set is =  0.2474794

    ## The accuracy of the CART model is =  0.9028414

    ## The accuracy of the Random Forest model with the best parameter mtry is =  0.9743355

    ## The accuracy of the Random Forest model with a default value for mtry parameter is =  0.9706691

    ## The accuracy of the boosting model is =  0.9706691

Which one would you recommend for this problem? Is your choice dierent from the model you recommended in part (b)? Why or why not?

    ## we choose the model Random Forest since it has the highest accuracy

    ##     Note: we need to keep in mind that using the accuracy as a metric     for the predictive performance of the models in this case is legitimate
    ##     because the raing set is balanced in the 4 possible classes. If we
    ##     had an unbalanced data set,we would have used other metrics like FPR,
    ##     FNR, Gini coefficient, AUC,.. to comare between models

    ## we note the random forest are still the best model for this problem, both when the predicted output is binary (two classes 'B' or 'not B'),and when the predicted output is multinomial 'A,'B', 'P', or 'R' . May be we can conlude that a model is not sensitive to the number of classes in a calssification problem: whether it is binary or multiclass, the predictive model has sensitively the same prediction performance
