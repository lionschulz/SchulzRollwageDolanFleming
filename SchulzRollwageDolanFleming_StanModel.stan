data {
  // specifying general data attributes
  int Participants;                         // number of participants
  int Trials;                               // number of trials per participant

  // Specifying the behavioural data that goes into the model (Always one column per participant and one row per trial)
  int SeeAgain[Trials, Participants];       // matrix of the binary decision to see again (note that this needs to be int instead of matrix to work with the bernoulli_logit() function)
  matrix[Trials, Participants] Confidence;  // matrix of the confidence reports
  matrix[Trials, Participants] Cost;        // matrix of the cost
  
  // Specifying the interindidual difference data
  vector[Participants] Dogmatism;
}
parameters {
  
  // INTERINDIVIDUAL DIFFERENCE PARAMETERS
  real B0DogEmbed;   // This variable let's the mean of the prior to vary as a function of dogmatism
  real B1DogEmbed;   // This variable let's the mean of the prior to vary as a function of dogmatism
  real B2DogEmbed;   // This variable let's the mean of the prior to vary as a function of dogmatism

  // PARAMETERS FOR THE LOGISTIC MODEL
  
  // Group level parameters, i.e. priors
  real Beta0_m;
  real <lower=0> Beta0_sd;
  real Beta1_m;
  real <lower=0>Beta1_sd;
  real Beta2_m;
  real <lower=0> Beta2_sd;
  
  // Individual level parameters
  real Beta0Individual[Participants];
  real Beta1Individual[Participants];
  real Beta2Individual[Participants];
  
}
model {
  // Specifying the group level parameters - we allow for a lot of leeway here
  Beta0_m~normal(0, 10);
  Beta0_sd~uniform(0, 10);
  Beta1_m~normal(0, 10);
  Beta1_sd~uniform(0, 10);
  Beta2_m~normal(0, 10);
  Beta2_sd~uniform(0, 10);
  
  
  // We specify the parametrisation for the individual paramters
  for (p in 1:Participants){
    
    // This means that the parameters are informed by the priors
    Beta0Individual[p] ~ normal(Beta0_m + B0DogEmbed*Dogmatism[p], Beta0_sd);
    Beta1Individual[p] ~ normal(Beta1_m + B1DogEmbed*Dogmatism[p], Beta1_sd);
    Beta2Individual[p] ~ normal(Beta2_m + B2DogEmbed*Dogmatism[p], Beta2_sd);
    
    // now we specify the model actual model, this is done on a trial by trial level
    for (t in 1:Trials) {
      
      // We specify the actual model
      if (Confidence[t, p] != 10){ // To avoid missed trials, we set up this if-function
      
        // this is the actual logistic model
        SeeAgain[t,p] ~ bernoulli_logit(Beta0Individual[p] + Beta1Individual[p]*Confidence[t,p] + Beta2Individual[p]*Cost[t,p]);
      
      }
    }
  }
}
