#--packages used--------------------------------------
require(rms)
require(lme4)
require(languageR)
require(multcomp)
require(lsmeans)
require(multcompView)
require(pbkrtest)
require(lmerTest)
require(randomForest)
require(party)
require(MASS)
library(car)
require(mgcv)
require(itsadug)
require(methods)
require(ggplot2)
library(devtools)
library(RePsychLing)

#---Dataset, design and variable info---------------------
# Full dataset
dat=read.table("Misic.EXP2.fulldatabase.txt",sep="\t",header=TRUE)

# subject_ID -- ID for each subject
# trial_ID -- ID for each stimulus
# trial_order -- position of a stimulus in a randomised presentation order
# item -- presented word
# modality -- experimental condition (auditory / visual LDT)
# word_length -- word length in letters
# familiarity -- word familiarity (1 - 7)*
# concreteness -- word concreteness (1 - 7)*
# frequency -- word frequency (per 2 million)**
# colthartN -- orthographic neighboorhood size
# entropy -- entropy of a word probability distribution**
# correct -- correct response
# RT -- reaction time in milliseconds
# auditory_stimulus_duration -- duration of the autditory stimulus, should be added to RT
# for auditory LDT task processing time
dat$RTos = dat$RT  
dat$RT = dat$RT + dat$auditory_stimulus_duration

# * taken from Serbian polysemy norms
# Đurđević, D. F., &  Kostić, A.
# Number, Relative Frequency, Entropy, Redundancy, Familiarity,
# and Concreteness of Word Senses: Ratings for
# 150 Serbian Polysemous Nouns. STUDIES IN LANGUAGE AND MIND 2, 13.

# ** taken from the Frequency Dictionary of Serbian language
# Kostić, Đ. (1999). Frekvencijski rečnik savremenog srpskog jezika.
# Tom I – VII. Institut za eksperimentalnu fonetiku i patologiju
# govora, Beograd i Laboratorija za eksperimentalnu psihologiju
# Filozofskog fakulteta u Beogradu

# 150 polysemous items 
# 143 participants
#   71 in visual LDT condition
#   72 in auditory LDT condition

#--DATA PREPROCESSING------------------------------------

#--removing participants with VLD error rates (<25%)----
sort(tapply(dat$correct, dat$subject_ID, sum)/150)

# no participants meet the criterion of
# 25% or more errors on word stimuli

#--removing stimuli with VLD error rates (<25%)---------
sort(tapply(dat$correct, dat$item, sum)/143)

# PISAK     SFERA     
# 0.3216783 0.6923077

dat = dat[dat$item!="PISAK",]
dat = dat[dat$item!="SFERA",]

# removing < 200 ms responses
dat = dat[dat$RT>200,]

#--removing incorrect responses-------------------------
# skip for accuracy analysis
dat = dat[dat$correct>0,]

#--RT transformation------------------------------------

# whole dataset
powerTransform(dat$RT)

# by condition
# modality: VLD
powerTransform(dat[dat$modality=="VLD",]$RT)

# modality: ALD
powerTransform(dat[dat$modality=="ALD",]$RT)

#inverse transformation
dat$invRT = -1000/dat$RT

#--plotting transformed RT-------------------------------

# whole dataset
par(mfrow=c(2,2))		
plot(sort(dat$invRT))
plot(density(dat$invRT))
qqnorm(dat$invRT)
par(mfrow=c(1,1))

#by condition
# modality: ALD
par(mfrow=c(2,2))		
plot(sort(dat[dat$modality == "ALD",]$invRT))
plot(density(dat[dat$modality == "ALD",]$invRT))
qqnorm(dat[dat$modality == "ALD",]$invRT)
par(mfrow=c(1,1))

# modality: VLD
par(mfrow=c(2,2))		
plot(sort(dat[dat$modality == "VLD",]$invRT))
plot(density(dat[dat$modality == "VLD",]$invRT))
qqnorm(dat[dat$modality == "VLD",]$invRT)
par(mfrow=c(1,1))

# whole dataset
ks.test(jitter(dat$invRT),"pnorm",mean(dat$invRT),sd(dat$invRT))

# modality: ALD
ks.test(jitter(dat[dat$modality == "ALD",]$invRT),
        "pnorm",mean(dat[dat$modality == "ALD",]$invRT),
        sd(dat[dat$modality == "ALD",]$invRT))

# modality: VLD
ks.test(jitter(dat[dat$modality == "VLD",]$invRT),
        "pnorm",mean(dat[dat$modality == "VLD",]$invRT),
        sd(dat[dat$modality == "VLD",]$invRT))

#--Scaling other predictors-------------------------------
# Scaling other predictors
dat$log_frequency=log(dat$frequency)
dat$log_frequency.z = scale(dat$log_frequency)

dat$trial_order.z = scale(dat$trial_order)
dat$word_length.z = scale(dat$word_length)
dat$familiarity.z = scale(dat$familiarity)
dat$concreteness.z = scale(dat$concreteness)
dat$colthartN.z = scale(dat$colthartN)
dat$entropy.z = scale(dat$entropy)

as.factor(as.character(dat$subject_ID))
as.factor(as.character(dat$item))

#--writing a preprocessed dataset------------------------
write.table(dat,"Misic.EXP2.preprocessed.txt",sep="\t",row.names=FALSE)
#--to skip preprocessing, start here---------------------
dat=read.table("Misic.EXP2.preprocessed.txt",sep="\t",header=TRUE)

#--DATA ANALYSIS-----------------------------------------

dat$modality <- relevel(dat$modality, ref = "VLD")

M1 <- lmer(invRT ~ 1 + trial_order.z + 
           modality * entropy.z +
           familiarity.z + word_length.z +
           (1 + familiarity.z + word_length.z + trial_order.z||subject_ID) +
           (1 + modality|item),
      data=dat, REML=FALSE)

M1.t <- lmer(invRT ~ 1 + trial_order.z + 
            modality * entropy.z +
            familiarity.z + word_length.z +
            (1 + familiarity.z + word_length.z + trial_order.z||subject_ID) +
            (1 + modality|item),
        data=dat, subset=abs(scale(resid(M1)))<2.5, REML=FALSE)

summary(rePCA(M1.t))

# $item
# Importance of components:
#   [,1]    [,2]
# Standard deviation     0.6462 0.21088
# Proportion of Variance 0.9038 0.09625
# Cumulative Proportion  0.9038 1.00000
# 
# $subject_ID
# Importance of components:
#   [,1]    [,2]    [,3]   [,4]
# Standard deviation     0.6884 0.18947 0.06470 0.0482
# Proportion of Variance 0.9179 0.06953 0.00811 0.0045
# Cumulative Proportion  0.9179 0.98739 0.99550 1.0000

summary(M1.t)
# AIC      BIC   logLik deviance df.resid 
# -10270.5 -10152.2   5150.3 -10300.5    19667 
# 
# Scaled residuals: 
#   Min      1Q  Median      3Q     Max 
# -3.0106 -0.5931 -0.0599  0.5409  3.2387 
# 
# Random effects:
#   Groups       Name          Variance  Std.Dev. Corr 
# item         (Intercept)   6.631e-03 0.081431      
# modalityALD   8.124e-03 0.090133 -0.81
# subject_ID   trial_order.z 1.146e-03 0.033857      
# subject_ID.1 word_length.z 1.337e-04 0.011561      
# subject_ID.2 familiarity.z 7.418e-05 0.008613      
# subject_ID.3 (Intercept)   1.513e-02 0.123017      
# Residual                   3.193e-02 0.178698      
# Number of obs: 19682, groups:  item, 147; subject_ID, 143
# 
# Fixed effects:
#   Estimate Std. Error         df  t value Pr(>|t|)    
# (Intercept)            -1.625254   0.016176 198.463819 -100.470  < 2e-16 ***
#   trial_order.z           0.009650   0.003110 140.325231    3.103 0.002319 ** 
#   modalityALD             0.620297   0.022027 178.260970   28.161  < 2e-16 ***
#   entropy.z              -0.024567   0.006971 140.884402   -3.524 0.000573 ***
#   familiarity.z          -0.033761   0.004354 149.402894   -7.755 1.27e-12 ***
#   word_length.z           0.015328   0.004378 155.973866    3.501 0.000605 ***
#   modalityALD:entropy.z   0.021676   0.007842 145.112877    2.764 0.006446 ** 
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Correlation of Fixed Effects:
#   (Intr) trl_r. mdlALD entrp. fmlrt. wrd_l.
# trial_rdr.z  0.001                                   
# modalityALD -0.721 -0.001                            
# entropy.z    0.001 -0.001 -0.001                     
# familirty.z  0.001  0.000  0.000 -0.075              
# wrd_lngth.z  0.001 -0.001  0.000  0.026  0.149       
# mdltyALD:n. -0.001  0.000  0.001 -0.794  0.002 -0.001

