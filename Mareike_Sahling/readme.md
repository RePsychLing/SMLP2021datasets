RMD file *Simdat.Rmd*: Data simulation for a binomial dependent
variable, analysis with LMM

**Experiment**:

-   XAB phoneme discrimination task (DV: categorization accuracy)

    -   Task: child is presented with three non-words (X, A and B) and
        has to judge whether X is more like A or B
    -   For every trial, the response is either correct or incorrect

-   80 children, two groups: control group and children with a history
    of Otitis Media with Effusion (OME group)
-   50 trials

    -   40 critical trials (split evenly between voicing trials and
        place of articulation trials)
    -   10 filler trials

-   RQ1: Overall categorization accuracy OME group \< controls ?
-   RQ 2: Interaction between group (OME vs. control) and contrast
    (voicing vs. place of articulation trials)?
-   Models:

    -   Fixed factors:

        -   group (OME vs. control)
        -   contrast (voicing vs. place of articulation)
        -   age (covariate)

    -   Random factors:

        -   subject
        -   item

**Questions**:

-   Any help regarding the simplification / improvement of the code and
    models is greatly appreciated
-   What possibilities are there to include a power analysis for the
    group comparison and interaction? The journal requires one and I am
    unsure how to go about this task as there are so many factors
    involved

    -   For the simulation, I worked with fixed effect sizes just to
        have a starting point but technically, my idea is that I would
        need to loop through the simulation with a range of possible
        effect sizes for the different factors and compute power for
        each? Is there another way?
