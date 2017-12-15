#############################################################################
############ He, Levine, Fan, Beemer, & Stronach    #########################
############ Random Forest as a Predictive Analytics#########################
############ Alternative in Institutional Research  #########################
#############################################################################
# December 14, 2017
# There are four pieces of R codes for this paper. 
# Subgroup analysis R code: 
#   The R code below was used to conduct the subgroup analysis (large p - scenario 1, Section 3.1, Figure 1, Table 1).
# Simulation R code: 
#   The R code below was used to conduct the simulation study on variable importance (Section 3.2, Table 1, Figure 2-4).
# Interpretation case analysis R code: 
#   The R code below was used to conduct the analysis (large p -scenario 2, Section 3.3, Figure 5, Table 2).
# Imbalanced data analysis R code: 
#   The R code below was used to conduct real case analysis on imbalanced outcome data (Section 3.4, Figure 7, Table 3).
# The single tree visualization in the paper was conducted in the R package rattle, code at bottom of this file
###

# R libraries required
library(caret);library(party);library(randomForest);library(ROCR);library(plyr)

### read in the compiled dataset and save in "data.m"
data.m$outcome.level=ifelse(data.m$grad4==0,"No","Yes");data.m$outcome.level=as.factor(data.m$outcome.level)
table(data.m$outcome.level)
#No Yes 
#916 336 

set.seed(88)
folds=createFolds(factor(data.m$outcome.level), k = 10, list = FALSE)

pred.accu.rf=NULL; sensitivity.rf=NULL;specificity.rf=NULL;time.rf=NULL;
pred.accu.CART=NULL;sensitivity.CART=NULL;specificity.CART=NULL; time.CART=NULL;
pred.accu.lr=NULL; sensitivity.lr=NULL;specificity.lr=NULL; time.lr=NULL;

pred.rf=list();pred.CART=list();pred.lr=list();
label.rf=list();label.CART=list();label.lr=list();

n.fold=10
for(i in 1:n.fold){
  data.m.test=data.m[folds==i, ]
  data.m.train=data.m[folds!=i, ]
  
  ##############
  ### CART #####
  ##############
  
  t0=proc.time()
  fit.cart=ctree(outcome.level ~., data=data.m.train) #pruning is not required
  t1=proc.time()-t0
  time.CART[i]=t1[3]
  a=table(predict(fit.cart,newdata=data.m.test[,-which(names(data.m.test)%in%"outcome.level")]), data.m.test$outcome.level)
  
  pred.accu.CART[i]=(a[1,1]+a[2,2])/sum(a)
  sensitivity.CART[i]=a[1,1]/sum(a[,1])
  specificity.CART[i]=a[2,2]/sum(a[,2])  
  pred=predict(fit.cart,newdata=data.m.test[,-which(names(data.m.test)%in%"outcome.level")],type="prob")
  predd=NULL
  for(j in 1:dim(data.m.test)[1]){
    predd=rbind(predd,pred[[j]])
  }
  pred.CART[[i]]=predd
  label.CART[[i]]=data.m.test$outcome.level
  
  ##############
  ##### RF #####
  ##############
  t0=proc.time()
  fit.rf=randomForest(outcome.level ~ .,data=data.m.train,ntree=500)
  t1=proc.time()-t0
  time.rf[i]=t1[3]
  
  b=table(predict(fit.rf,newdata=data.m.test[,-which(names(data.m.test)%in%"outcome.level")]), data.m.test$outcome.level)
  
  pred.accu.rf[i]=(b[1,1]+b[2,2])/sum(b)
  sensitivity.rf[i]=b[1,1]/sum(b[,1])
  specificity.rf[i]=b[2,2]/sum(b[,2])  
  pred.rf[[i]]=predict(fit.rf,newdata=data.m.test[,-which(names(data.m.test)%in%"outcome.level")],type="prob")
  label.rf[[i]]=data.m.test$outcome.level
  
  ##############
  ##### LR #####
  ##############
  t0=proc.time()
  
  ##### logistic regression - no VIF values #####
  no.contrast.v=NULL
  for(v in 1:length(names(data.m.train))){
    if(is.factor(data.m.train[,v])){
      no.contrast.v[v]=ifelse(length(unique(data.m.train[,v]))!=1,0,1)
    }
  }
  
  no.c.v=na.omit(names(data.m.train)[no.contrast.v==1])
  
  fit.lr=glm(outcome.level~., data=data.m.train[,!names(data.m.train)%in%no.c.v],family="binomial",control = list(maxit = 30))
  
  #the linearly dependent variables
  ld.vars=attributes(alias(fit.lr)$Complete)$dimnames[[1]]
  #remove the linearly dependent variables
  ld.vars=c("enrstat_des","admbas_des","newstd_des","sims_major_2","stem_2",
            "Initial.Grades.PHYS196","Initial.Grades.PHYS195", "Initial.Grades.MATH151" )
  
  fit.lr=glm(outcome.level~., data=data.m.train[,-which(names(data.m.train)%in%c(ld.vars,drop.v,no.c.v))],family="binomial",control = list(maxit = 30))
  fit.lr2=step(fit.lr)
  
  t1=proc.time()-t0
  #matching training and testing
  for (j in 1:length(names(data.m))){
    data.m.test=data.m.test[data.m.test[,j]%in%unique(data.m.train[,j]),]
  }
  
  c=table(ifelse(predict(fit.lr2,newdata=data.m.test[,-which(names(data.m.test)%in%c(c(ld.vars,drop.v),"outcome.level"))],type="response")<0.5,"No","Yes"), data.m.test$outcome.level)
  
  pred.accu.lr[i]=(c[1,1]+c[2,2])/sum(c)
  sensitivity.lr[i]=c[1,1]/sum(c[,1])
  specificity.lr[i]=c[2,2]/sum(c[,2])  
  time.lr[i]=t1[3]
  pred.lr[[i]]=predict(fit.lr2,newdata=data.m.test[,-which(names(data.m.test)%in%c(c(ld.vars,drop.v),"outcome.level"))],type="response")
  label.lr[[i]]=data.m.test$outcome.level

}

result=rbind.data.frame(c(mean(pred.accu.CART),mean(sensitivity.CART),mean(specificity.CART),mean(time.CART)),
                        c(mean(pred.accu.rf),mean(sensitivity.rf),mean(specificity.rf),mean(time.rf)),
                        c(mean(pred.accu.lr),mean(sensitivity.lr),mean(specificity.lr),mean(time.lr)))
names(result)=c("Accuracy","Sensitivity","Specificity","Comp time")
rownames(result)=c("CART","RF","LR")

# Table 2 results as an example
#       Accuracy Sensitivity Specificity Comp time
#CART  0.9168929   0.9781773   0.7503565     0.418
#RF    0.9185313   0.9432394   0.8514260     0.853
#LR    0.8646757   0.8963900   0.7734995  1241.223

###########################
#### CART tree plot #######
###########################

#### Decision tree can be plotted by R package "tree", "party", or "rpart" ####
library(tree)
fit.cart=tree(outcome.level ~., data=data.m)
plot(fit.cart, main="Classification Tree for ECE Students 4-year graduation success in STEM")
text(fit.cart)

#summary(fit.cart)
#cvTree=cv.tree(fit.cart, FUN = prune.misclass)  # run the cross validation
#plot(cvTree)  # plot the CV
treePrunedMod=prune.misclass(fit.cart, best =10 ) # set size corresponding to lowest value in below plot. try 4 or 16.
plot(treePrunedMod, main="Classification Tree for ECE Students 4-year graduation success in STEM - optimal subtree with 10 terminal nodes")
text(treePrunedMod)

library(party)
fit.cart=ctree(outcome.level ~., data=data.m)
plot(fit.cart, main="Classification Tree for ECE Students 4-year graduation success in STEM")

library(rpart);par(mar=c(0.5,0.5,0.5,0.5))
fit.cart=rpart(outcome.level ~., method="class",data=data.m, control=rpart.control(minsplit=2, minbucket=1, cp=0.001 ))
plot(fit.cart, uniform=TRUE)
text(fit.cart, use.n=TRUE, all=TRUE, cex=.8)
pfit.cart=prune(fit.cart, cp=fit.cart$cptable[which.min(fit.cart$cptable[,"xerror"]),"CP"])
plot(pfit.cart, uniform=TRUE)
text(pfit.cart, use.n=TRUE, all=TRUE, cex=.8)

##### The tree visualization used in the manuscript is generated through R package "rattle"
install.packages("rattle", dependencies=c("Depends", "Suggests"))
library(rattle)
rattle()














